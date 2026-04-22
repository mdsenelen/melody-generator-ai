export default function Footer() {
  return (
    <footer className="mt-auto w-full pb-6 pt-2 text-center text-[12px] text-white/30">
      <span>
        &copy; {new Date().getFullYear()} Merve Deniz Senelen
        {" · "}
        <a
          href="https://www.linkedin.com/in/mdenizsenelen"
          target="_blank"
          rel="noopener noreferrer"
          className="transition hover:text-white/60"
        >
          LinkedIn
        </a>
        {" · "}
        <a
          href="https://www.youtube.com/@mdsenelen"
          target="_blank"
          rel="noopener noreferrer"
          className="transition hover:text-white/60"
        >
          YouTube
        </a>
      </span>
    </footer>
  );
}
