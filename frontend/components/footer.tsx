export default function Footer() {
  return (
    <footer className="border-border text-muted-foreground font-title mt-auto w-full border-t py-6 text-center text-sm">
      <span>
        &copy; {new Date().getFullYear()} Merve Deniz Senelen
        {" · "}
        <a
          href="https://www.linkedin.com/in/mdenizsenelen"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-foreground transition-colors"
        >
          LinkedIn
        </a>
        {" · "}
        <a
          href="https://github.com/mdsenelen"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-foreground transition-colors"
        >
          Github
        </a>
      </span>
    </footer>
  );
}
