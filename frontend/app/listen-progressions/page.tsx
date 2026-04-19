import { ChordGraph } from "../../components/chord-graph";

const presetProgressions = [
  {
    title: "Pop",
    // description: "A bright loop for quick demos and topline sketching.",
    progression: ["C", "G", "Am", "F"],
  },
  {
    title: "Soul",
    // description: "A jazzy turnaround with softer harmony movement.",
    progression: ["Dm7", "G7", "Cmaj7", "Am7"],
  },
  {
    title: "Indie",
    // description: "A guitar-friendly progression with stronger push and release.",
    progression: ["Em", "C", "G", "D"],
  },
];

export default function ListenProgressionsPage() {
  return (
    <div className="space-y-6">
      <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
        {/* <h1 className="text-3xl font-semibold text-white" style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}>
          
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-white/75">
          
        </p> */}
      </section>
      <div className="grid gap-6">
        {presetProgressions.map((progression) => (
          <ChordGraph
            key={progression.title}
            title={progression.title}
            // description={progression.description}
            progression={progression.progression}
          />
        ))}
      </div>
    </div>
  );
}

