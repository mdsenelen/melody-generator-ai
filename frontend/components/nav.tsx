import React from "react";
import Link from "next/link";
import '../styles/nav.css';

const Nav = () => (
  <nav className="nav-inline, flex-row size-sm:justify-between">
    <Link href="/listen your chord progressions">Listen your chord progressions</Link>
    <Link href="/choose a chord progression to listen">Choose a chord progression to listen</Link>
    <Link href="/guide">User's Guide</Link>
  </nav>
);

export default Nav;