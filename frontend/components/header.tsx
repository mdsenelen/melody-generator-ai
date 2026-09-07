import Link from "next/link";

export default function Header() {
  return (
    <Link href="/" className="flex items-center gap-2.5 transition-opacity hover:opacity-75">
      <div className="flex h-7 w-7 items-center justify-center rounded-full border border-[#b9a1ff]/60 bg-[#8b5cf6]/20 text-[10px] font-bold text-[#f1d8ff] shadow-[0_0_12px_rgba(139,92,246,0.28)]">
        ✦
      </div>
      <span
        className="text-[17px] font-semibold tracking-[-0.04em] text-[#f5f0f7]"
        style={{ fontFamily: "var(--font-display)" }}
      >
        Melodia
      </span>
    </Link>
  );
}
