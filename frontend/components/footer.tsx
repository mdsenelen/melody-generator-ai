export default function Footer() {
  return (
    <footer className="border-border text-muted-foreground mt-auto w-full border-t py-6 text-center font-mono text-[11px] tracking-wide">
      <span>
        &copy; {new Date().getFullYear()} Merve Deniz Senelen
        {" · "}
        <a
          href="https://www.linkedin.com/in/mdenizsenelen"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-primary transition-colors"
        >
          LinkedIn
        </a>
        {" · "}
        <a
          href="https://www.youtube.com/@mdsenelen"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-primary transition-colors"
        >
          YouTube
        </a>
      </span>
    </footer>
  );
}
