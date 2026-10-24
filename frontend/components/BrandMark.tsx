/** The aperture keeps the measured center open; its exit links evidence to an answer. */
export default function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="currentColor"
      className="brand-symbol"
      aria-hidden="true"
      focusable="false"
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M4 4h16l8 8v16H12l-8-8V4Zm6 6v8l4 4h8v-8l-4-4h-8Z"
      />
      <path d="m18 18 12 12h-8l-8-8 4-4Z" />
    </svg>
  );
}
