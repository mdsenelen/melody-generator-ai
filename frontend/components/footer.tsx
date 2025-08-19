import React from "react";

export const Footer = () => {
  return (
    <footer className="mt-auto w-full py-4 text-center text-[0.8rem] text-text-secondary dark:text-d-text-secondary">
      <span>
        &copy; {new Date().getFullYear()} Merve Deniz Senelen. Find me on{" "}
        <a
          href="https://www.linkedin.com/in/mdenizsenelen"
          target="_blank"
          rel="noopener noreferrer"
        >
          LinkedIn
        </a>{" "}
        or{" "}
        <a
          href="https://www.youtube.com/@mdsenelen"
          target="_blank"
          rel="noopener noreferrer"
        >
          YouTube
        </a>
        .
        .
      </span>
    </footer>
  );
};

export default Footer;
