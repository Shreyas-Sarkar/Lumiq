-- Additive migration. Audit duplicates before creating the unique index; never delete history.
do $$ begin
 if exists(select 1 from public.dataset group by chat_id having count(*)>1) then
   raise exception 'Multiple datasets per chat exist. Audit them before this migration.';
 end if;
end $$;
create unique index if not exists dataset_one_per_chat on public.dataset(chat_id);
-- Paths/profiles are server-authored. Owner RLS alone would let a client point
-- its dataset row at somebody else's storage object using the service-role reader.
revoke insert,update,delete on public.dataset from anon,authenticated;
grant select on public.dataset to authenticated;
alter table public.dataset add column if not exists content_hash text;
alter table public.dataset add column if not exists profile jsonb not null default '{}';

create table if not exists public.investigation_run (
 id uuid primary key default gen_random_uuid(),
 user_id uuid not null references public.users(id) on delete cascade,
 chat_id uuid not null references public.chat(id) on delete cascade,
 dataset_id uuid not null references public.dataset(id) on delete cascade,
 investigation_id uuid not null,
 request_id text not null,
 revision integer not null default 0,
 state jsonb not null default '{}',
 lease_token uuid,
 lease_until timestamptz,
 created_at timestamptz not null default now(),
 updated_at timestamptz not null default now(),
 unique(user_id, request_id)
);
create index if not exists investigation_owner_chat on public.investigation_run(user_id,chat_id,created_at desc);
alter table public.investigation_run enable row level security;
-- Clients can read their runs but only the backend may change analytical state.
create policy investigation_owner_read on public.investigation_run for select to authenticated
 using ((select auth.uid())=user_id);
revoke all on public.investigation_run from authenticated;
grant select on public.investigation_run to authenticated;
grant all on public.investigation_run to service_role;
revoke all on public.investigation_run from anon;

create table if not exists public.provider_budget (
 provider text primary key,
 cooldown_until timestamptz,
 disabled_reason text,
 window_start timestamptz not null default now(),
 minute_calls integer not null default 0,
 minute_tokens integer not null default 0,
 day_key text not null default '',
 day_calls integer not null default 0
);
alter table public.provider_budget enable row level security;
revoke all on public.provider_budget from anon, authenticated;
grant all on public.provider_budget to service_role;

-- Global compute lease prevents multiple Render workers from exceeding the small memory budget.
create table if not exists public.compute_lease (
 id integer primary key check(id=1), token uuid, expires_at timestamptz
);
insert into public.compute_lease(id) values(1) on conflict do nothing;
alter table public.compute_lease enable row level security;
revoke all on public.compute_lease from anon,authenticated;
grant all on public.compute_lease to service_role;

create or replace function public.claim_investigation(p_id uuid,p_owner uuid,p_token uuid)
returns setof public.investigation_run language plpgsql security invoker set search_path='' as $$
begin
 perform pg_advisory_xact_lock(981237);
 if exists(select 1 from public.investigation_run where user_id=p_owner and id<>p_id and lease_until>now()) then return; end if;
 if not exists(select 1 from public.investigation_run where id=p_id and user_id=p_owner and (lease_until is null or lease_until<now())) then return; end if;
 update public.compute_lease set token=p_token,expires_at=now()+interval '120 seconds'
 where id=1 and (expires_at is null or expires_at<now());
 if not found then return; end if;
 return query update public.investigation_run set lease_token=p_token,lease_until=now()+interval '120 seconds'
 where id=p_id and user_id=p_owner and (lease_until is null or lease_until<now()) returning *;
end $$;

create or replace function public.checkpoint_investigation(p_id uuid,p_owner uuid,p_token uuid,p_state jsonb,p_release boolean)
returns setof public.investigation_run language plpgsql security invoker set search_path='' as $$
begin
 return query update public.investigation_run set state=p_state, revision=revision+1, updated_at=now(),
 lease_until=case when p_release then null else now()+interval '120 seconds' end,
 lease_token=case when p_release then null else p_token end
 where id=p_id and user_id=p_owner and lease_token=p_token and lease_until>now() returning *;
 if not found then raise exception 'Run lease lost'; end if;
 update public.compute_lease set token=case when p_release then null else p_token end,
 expires_at=case when p_release then null else now()+interval '120 seconds' end where id=1 and token=p_token;
end $$;

-- Atomic reservations. Limits are configured from the account quota, not assumed from the model name.
create or replace function public.reserve_provider(p_provider text,p_day text,p_rpm int,p_rpd int,p_tpm int,p_tokens int)
returns jsonb language plpgsql security invoker set search_path='' as $$
declare b public.provider_budget;
begin
 insert into public.provider_budget(provider) values(p_provider) on conflict do nothing;
 select * into b from public.provider_budget where provider=p_provider for update;
 if b.disabled_reason is not null then return jsonb_build_object('allowed',false,'reason',b.disabled_reason); end if;
 if b.cooldown_until>now() then return jsonb_build_object('allowed',false,'reason','quota','retry_at',b.cooldown_until); end if;
 if b.window_start<now()-interval '60 seconds' then b.window_start=now();b.minute_calls=0;b.minute_tokens=0; end if;
 if b.day_key<>p_day then b.day_key=p_day;b.day_calls=0; end if;
 if b.day_calls>=p_rpd then return jsonb_build_object('allowed',false,'reason','daily_quota'); end if;
 if b.minute_calls>=p_rpm or b.minute_tokens+p_tokens>p_tpm then
   return jsonb_build_object('allowed',false,'reason','quota','retry_at',b.window_start+interval '60 seconds');
 end if;
 update public.provider_budget set window_start=b.window_start,minute_calls=b.minute_calls+1,minute_tokens=b.minute_tokens+p_tokens,day_key=b.day_key,day_calls=b.day_calls+1 where provider=p_provider;
 return jsonb_build_object('allowed',true);
end $$;

-- Only service_role may invoke RPCs; no SECURITY DEFINER bypass and no public mutation endpoint.
revoke all on function public.claim_investigation(uuid,uuid,uuid) from public,anon,authenticated;
revoke all on function public.checkpoint_investigation(uuid,uuid,uuid,jsonb,boolean) from public,anon,authenticated;
revoke all on function public.reserve_provider(text,text,int,int,int,int) from public,anon,authenticated;
grant execute on function public.claim_investigation(uuid,uuid,uuid) to service_role;
grant execute on function public.checkpoint_investigation(uuid,uuid,uuid,jsonb,boolean) to service_role;
grant execute on function public.reserve_provider(text,text,int,int,int,int) to service_role;

insert into storage.buckets(id,name,public,file_size_limit,allowed_mime_types)
 values('lumiq-datasets','lumiq-datasets',false,5242880,array['text/csv'])
 on conflict(id) do nothing;
-- Bucket is backend-only. The backend authenticates every read and derives paths from owned metadata.

do $$ begin
 if exists(select 1 from storage.buckets where id='lumiq-datasets' and public) then
  raise exception 'Existing lumiq-datasets bucket is public. Make it private before enabling uploads.';
 end if;
end $$;
