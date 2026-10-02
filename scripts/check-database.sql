-- Run against a disposable database after both migrations. Rolls back fixture data.
begin;
insert into auth.users(id,email) values('11111111-1111-4111-8111-111111111111','owner@example.test'),('44444444-4444-4444-8444-444444444444','other@example.test');
insert into public.chat(id,user_id) values('22222222-2222-4222-8222-222222222222','11111111-1111-4111-8111-111111111111');
insert into public.dataset(id,user_id,chat_id,filename,storage_path,chroma_collection_id) values('33333333-3333-4333-8333-333333333333','11111111-1111-4111-8111-111111111111','22222222-2222-4222-8222-222222222222','test.csv','supabase://lumiq-datasets/owner/test.csv','test');
insert into public.investigation_run(id,user_id,chat_id,dataset_id,investigation_id,request_id) values('55555555-5555-4555-8555-555555555555','11111111-1111-4111-8111-111111111111','22222222-2222-4222-8222-222222222222','33333333-3333-4333-8333-333333333333','55555555-5555-4555-8555-555555555555','test-request');
set local role service_role;
do $$ begin
 if (select count(*) from public.claim_investigation('55555555-5555-4555-8555-555555555555','44444444-4444-4444-8444-444444444444','66666666-6666-4666-8666-666666666666'))<>0 then raise exception 'Foreign owner acquired lease'; end if;
 if (select count(*) from public.claim_investigation('55555555-5555-4555-8555-555555555555','11111111-1111-4111-8111-111111111111','66666666-6666-4666-8666-666666666666'))<>1 then raise exception 'Owner failed lease'; end if;
 if (select count(*) from public.claim_investigation('55555555-5555-4555-8555-555555555555','11111111-1111-4111-8111-111111111111','77777777-7777-4777-8777-777777777777'))<>0 then raise exception 'Duplicate lease'; end if;
 if not ((public.reserve_provider('gemini','2026-10-02',1,2,12000,4000)->>'allowed')::boolean) then raise exception 'Reservation failed'; end if;
 if ((public.reserve_provider('gemini','2026-10-02',1,2,12000,4000)->>'allowed')::boolean) then raise exception 'Rate limit not enforced'; end if;
 perform public.checkpoint_investigation('55555555-5555-4555-8555-555555555555','11111111-1111-4111-8111-111111111111','66666666-6666-4666-8666-666666666666','{"status":"complete","evidence":[{"id":"e1"}]}',true);
end $$;
reset role;
set local role authenticated;
select set_config('request.jwt.claim.sub','44444444-4444-4444-8444-444444444444',true);
do $$ begin if (select count(*) from public.investigation_run)<>0 then raise exception 'Foreign run visible'; end if; end $$;
select set_config('request.jwt.claim.sub','11111111-1111-4111-8111-111111111111',true);
do $$ begin if (select count(*) from public.investigation_run)<>1 then raise exception 'Owner run missing'; end if; end $$;
reset role;
do $$ begin
 if has_function_privilege('authenticated','public.claim_investigation(uuid,uuid,uuid)','EXECUTE') then raise exception 'Client may acquire worker lease'; end if;
 if has_table_privilege('authenticated','public.provider_budget','SELECT') then raise exception 'Private budgets exposed'; end if;
end $$;
rollback;
