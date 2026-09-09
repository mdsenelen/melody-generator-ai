import Link from "next/link";

export default function Header() {
  return (
    <Link href="/" className="flex items-center gap-2.5 transition-opacity hover:opacity-75">
      <img src="/logo.svg" alt="" className="h-7 w-7 object-contain" aria-hidden="true" />
      <span
        className="text-[17px] font-semibold tracking-[-0.04em] text-[#f5f0f7]"
        style={{ fontFamily: "var(--font-display)" }}
      >
        Melodia
      </span>
    </Link>
  );
}
