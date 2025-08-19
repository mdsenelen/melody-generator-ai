import React, { useEffect, useState } from 'react';

function ChordDropdown() {
  const [chords, setChords] = useState<string[]>([]);

  useEffect(() => {
    fetch('http://localhost:8000/chords')
      .then(response => response.json())
      .then((data: { chords: string[] }) => {
        setChords(data.chords);
      });
  }, []);

  return (
    <select>
      {chords.map((chord, idx) => (
        <option key={idx} value={chord}>{chord}</option>
      ))}
    </select>
  );
}

export default ChordDropdown;