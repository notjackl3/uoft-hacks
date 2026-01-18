import React, { useState } from 'react';

interface Recommendation {
  product_id: string;
  title: string;
  vendor?: string;
  price: number;
  currency: string;
  url?: string;
  image_url?: string;
  health_score: number;
  score_improvement: number;
  why_healthier: string[];
  comparison_summary: string;
  confidence: string;
}

interface SearchResult {
  baseline: {
    title: string;
    health_score: number;
    factors: string[];
  };
  recommendations: Recommendation[];
  allergy_warning?: string;
  decision_ids: string[];
  session_id: string;
}

interface HealthySnacksProps {
  onBuyProduct?: (product: Recommendation) => void;
}

// Company ID with Shopify products - in production this would be configurable
const COMPANY_ID = 'f20b11b9-8252-44ea-93a9-a272615b1019';
const API_BASE = 'http://localhost:8000';

const HealthySnacks: React.FC<HealthySnacksProps> = ({ onBuyProduct }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [budget, setBudget] = useState<string>('');
  const [allergies, setAllergies] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult | null>(null);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      const requestBody: any = {
        baseline_description: searchQuery.trim(),
        limit: 5,
      };

      if (budget) {
        requestBody.budget = parseFloat(budget);
      }

      if (allergies.trim()) {
        requestBody.allergies = allergies.split(',').map(a => a.trim()).filter(Boolean);
      }

      const response = await fetch(`${API_BASE}/api/commerce/${COMPANY_ID}/recommendations/snacks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Failed to get recommendations: ${response.statusText}`);
      }

      const data: SearchResult = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to search');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="flex flex-col h-full">
      {/* Search Header */}
      <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-4">
        <h2 className="text-lg font-bold flex items-center gap-2">
          🥗 Healthy Snack Finder
        </h2>
        <p className="text-sm text-green-100 mt-1">
          Find healthier alternatives to your favorite snacks
        </p>
      </div>

      {/* Search Form */}
      <div className="bg-white border-b px-4 py-4 space-y-3">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            What snack do you want to replace?
          </label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="e.g., Doritos, Oreos, potato chips..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
          />
        </div>

        <div className="flex gap-3">
          <div className="flex-1">
            <label className="block text-xs font-medium text-gray-600 mb-1">
              💰 Max Budget (CAD)
            </label>
            <input
              type="number"
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
              placeholder="e.g., 8"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-green-500"
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs font-medium text-gray-600 mb-1">
              🚫 Allergies (comma-separated)
            </label>
            <input
              type="text"
              value={allergies}
              onChange={(e) => setAllergies(e.target.value)}
              placeholder="e.g., peanuts, dairy"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-green-500"
            />
          </div>
        </div>

        <button
          onClick={handleSearch}
          disabled={isLoading || !searchQuery.trim()}
          className="w-full py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? '🔍 Searching...' : '🔍 Find Healthier Alternatives'}
        </button>
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
            ❌ {error}
          </div>
        )}

        {results && (
          <div className="space-y-4">
            {/* Baseline Info */}
            <div className="bg-gray-50 rounded-lg p-3 border">
              <div className="text-sm text-gray-600">Searching for alternatives to:</div>
              <div className="font-medium text-gray-800">{results.baseline.title}</div>
              <div className="text-sm text-gray-500 mt-1">
                Health Score: <span className={getScoreColor(results.baseline.health_score)}>
                  {results.baseline.health_score}/100
                </span>
              </div>
            </div>

            {/* Allergy Warning */}
            {results.allergy_warning && (
              <div className="bg-orange-50 border border-orange-200 text-orange-700 px-3 py-2 rounded-lg text-sm">
                {results.allergy_warning}
              </div>
            )}

            {/* Recommendations */}
            <div className="text-sm font-medium text-gray-700">
              Found {results.recommendations.length} healthier alternatives:
            </div>

            {results.recommendations.map((rec, index) => (
              <div key={rec.product_id} className="bg-white border rounded-lg shadow-sm overflow-hidden">
                <div className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-lg font-medium text-gray-800">
                          {index + 1}. {rec.title}
                        </span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${getConfidenceColor(rec.confidence)}`}>
                          {rec.confidence}
                        </span>
                      </div>
                      {rec.vendor && (
                        <div className="text-sm text-gray-500">{rec.vendor}</div>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-green-600">
                        ${rec.price.toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-500">{rec.currency}</div>
                    </div>
                  </div>

                  {/* Health Score */}
                  <div className="mt-3 flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-600">Health Score:</span>
                      <span className={`font-bold ${getScoreColor(rec.health_score)}`}>
                        {rec.health_score}/100
                      </span>
                    </div>
                    <div className="text-sm text-green-600 font-medium">
                      +{rec.score_improvement} improvement
                    </div>
                  </div>

                  {/* Why Healthier */}
                  <div className="mt-3">
                    <div className="text-xs font-medium text-gray-600 mb-1">Why it's healthier:</div>
                    <div className="flex flex-wrap gap-1">
                      {rec.why_healthier.slice(0, 3).map((reason, i) => (
                        <span key={i} className="text-xs bg-green-50 text-green-700 px-2 py-1 rounded">
                          ✓ {reason}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Comparison */}
                  <div className="mt-2 text-sm text-gray-600 italic">
                    "{rec.comparison_summary}"
                  </div>

                  {/* Buy Button - triggers autonomous navigation */}
                  <button
                    onClick={() => onBuyProduct?.(rec)}
                    className="mt-3 block w-full text-center py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
                  >
                    🤖 Buy This For Me
                  </button>
                </div>
              </div>
            ))}

            {results.recommendations.length === 0 && (
              <div className="text-center text-gray-500 py-8">
                No healthier alternatives found matching your criteria.
                <br />
                Try adjusting your budget or allergies.
              </div>
            )}

            {/* Decision ID */}
            <div className="text-xs text-gray-400 text-center mt-4">
              Decision ID: {results.decision_ids[0]?.slice(0, 8)}...
            </div>
          </div>
        )}

        {!results && !error && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="text-4xl mb-4">🥗</div>
            <p className="text-center">
              Enter a snack above to find<br />healthier alternatives
            </p>
            <div className="mt-4 text-xs text-gray-400 max-w-xs text-center">
              Examples: "potato chips", "chocolate cookies", "candy bars"
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HealthySnacks;
