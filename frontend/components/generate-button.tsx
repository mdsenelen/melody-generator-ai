'use client';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type GenerateButtonProps = {
  filename: string;
  detectedChords?: string[];
  onGenerated: (generatedFile: string) => void;
};

export function GenerateButton({ filename, detectedChords = [], onGenerated }: GenerateButtonProps) {
  const [loading, setLoading] = useState(false);
  const [selectedChord, setSelectedChord] = useState<string>('');
  const [creativity, setCreativity] = useState(0.7);
  const [generationType, setGenerationType] = useState<'normal' | 'chord' | 'progression'>('normal');
  const [showOptions, setShowOptions] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      let endpoint = '/api/generate';
      let body: any = { filename };

      if (generationType === 'chord' && selectedChord) {
        body.chord = selectedChord;
        body.creativity = creativity;
      } else if (generationType === 'progression') {
        endpoint = '/api/generate-progression';
        body = {
          roman: ["I", "V", "vi", "IV"],
          key: "C",
          segment_seconds: 2.0,
          creativity: creativity
        };
      } else {
        body.creativity = creativity;
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error('Generation failed');

      const data = await res.json();
      onGenerated(data.filename || data.download_url?.split('/').pop());
    } catch (e) {
      console.error(e);
      alert("Error during generation");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-4xl mx-auto space-y-6"
    >
      {/* Main Generation Panel */}
      <div className="bg-gradient-to-br from-gray-900/80 to-gray-800/80 rounded-2xl p-8 border border-gray-700/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white flex items-center">
            <span className="mr-3">🎹</span>
            Generate Music
          </h2>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowOptions(!showOptions)}
            className="px-4 py-2 bg-purple-600/50 text-white rounded-lg hover:bg-purple-600 transition-colors"
          >
            {showOptions ? 'Hide Options' : 'Show Options'}
          </motion.button>
        </div>

        {/* Generation Type Selector */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {[
            { id: 'normal', label: '🎵 Normal Generation', desc: 'Generate from uploaded audio' },
            { id: 'chord', label: '🎸 Chord-Based', desc: 'Generate using detected chords' },
            { id: 'progression', label: '🎼 Progression', desc: 'Generate from chord progressions' }
          ].map((type) => (
            <motion.button
              key={type.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setGenerationType(type.id as any)}
              className={`
                p-4 rounded-xl border-2 transition-all duration-200 text-left
                ${generationType === type.id
                  ? 'border-purple-500 bg-purple-600/20 text-white'
                  : 'border-gray-600 bg-gray-700/30 text-gray-300 hover:border-gray-500'
                }
              `}
            >
              <div className="font-semibold mb-1">{type.label}</div>
              <div className="text-sm opacity-75">{type.desc}</div>
            </motion.button>
          ))}
        </div>

        {/* Advanced Options */}
        <AnimatePresence>
          {showOptions && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-6 border-t border-gray-700 pt-6"
            >
              {/* Creativity Slider */}
              <div>
                <label className="block text-white font-medium mb-2">
                  🎨 Creativity Level: {creativity}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={creativity}
                  onChange={(e) => setCreativity(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>Conservative</span>
                  <span>Balanced</span>
                  <span>Creative</span>
                </div>
              </div>

              {/* Chord Selection */}
              {generationType === 'chord' && detectedChords.length > 0 && (
                <div>
                  <label className="block text-white font-medium mb-3">
                    🎸 Select Base Chord:
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {detectedChords.map((chord) => (
                      <motion.button
                        key={chord}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setSelectedChord(chord)}
                        className={`
                          px-4 py-2 rounded-lg font-mono text-sm transition-all duration-200
                          ${selectedChord === chord
                            ? 'bg-purple-600 text-white shadow-lg'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                          }
                        `}
                      >
                        {chord}
                      </motion.button>
                    ))}
                  </div>
                </div>
              )}

              {/* Progression Options */}
              {generationType === 'progression' && (
                <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-500/30">
                  <h4 className="text-white font-medium mb-2">🎼 Popular Progressions</h4>
                  <p className="text-gray-400 text-sm mb-3">
                    Using I-V-vi-IV progression in C major
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {["C", "G", "Am", "F"].map((chord, index) => (
                      <span key={chord} className="px-3 py-1 bg-blue-600/50 text-white rounded font-mono text-sm">
                        {chord}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Generate Button */}
        <div className="mt-6">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleGenerate}
            disabled={loading || (generationType === 'chord' && !selectedChord)}
            className={`
              w-full py-4 px-6 rounded-xl font-bold text-lg transition-all duration-200
              ${loading
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-700 hover:to-emerald-700 shadow-lg hover:shadow-xl'
              }
            `}
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                Generating...
              </div>
            ) : (
              <div className="flex items-center justify-center">
                <span className="mr-2">✨</span>
                Generate Music
              </div>
            )}
          </motion.button>
        </div>
      </div>

      {/* Info Panel */}
      <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/30">
        <div className="flex items-start space-x-4">
          <div className="text-2xl">💡</div>
          <div>
            <h4 className="text-white font-medium mb-1">Generation Tips</h4>
            <p className="text-gray-400 text-sm">
              {generationType === 'normal' && 'Generate variations of your uploaded audio with AI'}
              {generationType === 'chord' && 'Create new music based on the detected chords from your audio'}
              {generationType === 'progression' && 'Generate music using popular chord progressions'}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}