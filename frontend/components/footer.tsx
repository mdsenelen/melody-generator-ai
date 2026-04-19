import React from "react";

export const Footer = () => {
  return (
    <footer className="mt-auto w-full pb-6 text-center text-[0.8rem] text-gray-400">
      <span>
        &copy; {new Date().getFullYear()} Merve Deniz Senelen. Find me on{" "}
        <a
          href="https://www.linkedin.com/in/mdenizsenelen"
          target="_blank"
          rel="noopener noreferrer"
          className="text-purple-200 transition hover:text-white"
        >
          LinkedIn
        </a>{" "}
        or{" "}
        <a
          href="https://www.youtube.com/@mdsenelen"
          target="_blank"
          rel="noopener noreferrer"
          className="text-purple-200 transition hover:text-white"
        >
          YouTube
        </a>
        .
      </span>
    </footer>
  );
};

export default Footer;
