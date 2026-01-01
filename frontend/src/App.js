import React, { useState } from 'react';
import { Sparkles, FileText, CheckCircle, AlertCircle, Zap } from 'lucide-react';

const AceternityDocClassifier = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setMousePosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const classifyDocument = async () => {
    if (text.length < 5) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, top_k: 3, threshold: 0.65 })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Classification error:', error);
      // Optionally show error to user
      alert('Failed to classify document. Please ensure the API is running.');
    }
    setLoading(false);
  };

  const exampleTexts = [
    { label: 'Invoice', text: 'Invoice #8472 Total Due $1,250.50. Payment due within 30 days.' },
    { label: 'Receipt', text: 'Receipt #3391 Thank you for your purchase. Total $45.99.' },
    { label: 'Contract', text: 'This agreement is made on 01/12/2023 between Party A and Party B.' },
    { label: 'ID', text: 'Driver License Number D7834. Date of Birth 05/03/1990.' }
  ];

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      {/* Animated gradient background */}
      <div className="fixed inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(120,0,255,0.1),transparent_50%)]" />
      </div>

      {/* Grid overlay */}
      <div className="fixed inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px]" />

      {/* Content */}
      <div className="relative z-10">
        {/* Header */}
        <header className="pt-20 pb-10 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 mb-6 bg-white/5 border border-white/10 rounded-full backdrop-blur-sm">
            <Sparkles className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-300">AI-Powered Classification</span>
          </div>
          
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent animate-gradient">
            Document Classifier
          </h1>
          
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Instantly classify your documents using advanced machine learning
          </p>
        </header>

        {/* Main content */}
        <main className="max-w-6xl mx-auto px-6 pb-20">
          <div className="grid md:grid-cols-2 gap-8">
            {/* Input section */}
            <div 
              className="relative group"
              onMouseMove={handleMouseMove}
            >
              <div 
                className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500"
                style={{
                  background: `radial-gradient(circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(168, 85, 247, 0.4), transparent 50%)`
                }}
              />
              
              <div className="relative bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <FileText className="w-5 h-5 text-purple-400" />
                  <h2 className="text-xl font-semibold">Enter Document Text</h2>
                </div>
                
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste your document text here..."
                  className="w-full h-64 bg-white/5 border border-white/10 rounded-xl p-4 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 transition resize-none"
                />
                
                <div className="flex gap-3 mt-4">
                  <button
                    onClick={classifyDocument}
                    disabled={text.length < 5 || loading}
                    className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-3 px-6 rounded-xl transition transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Zap className="w-5 h-5" />
                        Classify Document
                      </>
                    )}
                  </button>
                </div>

                {/* Example buttons */}
                <div className="mt-6 pt-6 border-t border-white/10">
                  <p className="text-sm text-gray-400 mb-3">Try examples:</p>
                  <div className="grid grid-cols-2 gap-2">
                    {exampleTexts.map((example, idx) => (
                      <button
                        key={idx}
                        onClick={() => setText(example.text)}
                        className="text-left text-sm bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg p-3 transition"
                      >
                        <span className="text-purple-400 font-semibold">{example.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Results section */}
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500" />
              
              <div className="relative bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-6 h-full">
                <h2 className="text-xl font-semibold mb-4">Classification Results</h2>
                
                {!result ? (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-purple-500/20 to-blue-500/20 rounded-2xl flex items-center justify-center">
                        <Sparkles className="w-8 h-8 text-purple-400" />
                      </div>
                      <p className="text-gray-400">Enter text and click classify to see results</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Main result */}
                    <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-xl p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <p className="text-sm text-gray-400 mb-1">Predicted Label</p>
                          <h3 className="text-3xl font-bold capitalize bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                            {result.label}
                          </h3>
                        </div>
                        {!result.abstained && (
                          <CheckCircle className="w-8 h-8 text-green-400" />
                        )}
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Confidence</span>
                          <span className="font-semibold text-white">{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-white/5 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full transition-all duration-1000"
                            style={{ width: `${result.confidence * 100}%` }}
                          />
                        </div>
                      </div>

                      {result.abstained && (
                        <div className="mt-4 flex items-center gap-2 text-yellow-400">
                          <AlertCircle className="w-4 h-4" />
                          <span className="text-sm">Low confidence - needs review</span>
                        </div>
                      )}
                    </div>

                    {/* Top predictions */}
                    <div>
                      <h4 className="text-sm font-semibold text-gray-400 mb-3">Top Predictions</h4>
                      <div className="space-y-2">
                        {result.top_k.map((item, idx) => (
                          <div 
                            key={idx}
                            className="bg-white/5 border border-white/10 rounded-lg p-4 hover:bg-white/10 transition"
                          >
                            <div className="flex justify-between items-center mb-2">
                              <span className="font-medium capitalize">{item.label}</span>
                              <span className="text-sm text-gray-400">{(item.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-gradient-to-r from-purple-500/50 to-blue-500/50 rounded-full"
                                style={{ width: `${item.confidence * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Features section */}
          <div className="grid md:grid-cols-3 gap-6 mt-16">
            {[
              { icon: Zap, title: 'Lightning Fast', desc: 'Get results in milliseconds' },
              { icon: CheckCircle, title: 'High Accuracy', desc: '92%+ classification accuracy' },
              { icon: Sparkles, title: 'Smart Abstention', desc: 'Knows when to ask for review' }
            ].map((feature, idx) => (
              <div 
                key={idx}
                className="relative group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 hover:bg-white/10 transition"
              >
                <feature.icon className="w-10 h-10 text-purple-400 mb-4" />
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-400 text-sm">{feature.desc}</p>
              </div>
            ))}
          </div>
        </main>
      </div>

      <style>{`
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient {
          background-size: 200% auto;
          animation: gradient 3s ease infinite;
        }
      `}</style>
    </div>
  );
};

export default AceternityDocClassifier;