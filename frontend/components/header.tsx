import LogoRotationVariant from "./logo-rotation";
import Link from "next/link";
import React from "react";
import '../styles/header.css'; // Add this import for custom styles

export default function Header() {
  return (
    <div className="header-80s">
      <Link href="/" className="flex items-center gap-4 hover:opacity-80 transition-opacity">
        <LogoRotationVariant
          src="/logo.svg"
          alt="Music VAE Logo" 
          width={35} 
          height={35}
          className="h-10 w-10"
        />
        <h1 className="font-mono text-xl items-center font-semibold tracking-widest text-white">
          Musical Playground
        </h1>
      </Link>
    </div>
  );
}