module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        melodia: {
          bg: "#070b14",
          bgAlt: "#0b1220",
          panel: "rgba(12, 18, 29, 0.78)",
          panelStrong: "rgba(16, 22, 34, 0.94)",
          border: "rgba(148, 163, 184, 0.18)",
          text: "#eef3ff",
          textSoft: "rgba(220, 227, 240, 0.78)",
          muted: "rgba(148, 163, 184, 0.8)",
          accent: "#8b5cf6",
          accentStrong: "#a78bfa",
          accent2: "#38bdf8",
          accent3: "#fbbf24",
          glow: "rgba(139, 92, 246, 0.26)",
        },
      },
      boxShadow: {
        melodia: "0 22px 60px rgba(2, 6, 23, 0.6)",
      },
    },
  },
};
