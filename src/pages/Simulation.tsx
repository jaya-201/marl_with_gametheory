import React, { useState } from 'react';
import { Play, Download, ChartBar as BarChart3 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { generateSimulationResults, formatCurrency } from '../utils/helpers';
import { AIRLINES } from '../utils/constants';

export const Simulation: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [episodes, setEpisodes] = useState(10);
  const [results, setResults] = useState<any[]>([]);
  const [progress, setProgress] = useState(0);

  const handleRunSimulation = () => {
    setIsRunning(true);
    setProgress(0);
    setResults([]);

    // Simulate running episodes
    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + (100 / episodes);
        if (newProgress >= 100) {
          clearInterval(interval);
          setIsRunning(false);
          setResults(generateSimulationResults(episodes));
          return 100;
        }
        return newProgress;
      });
    }, 200);
  };

  const exportResults = () => {
    if (results.length === 0) return;
    
    const csv = [
      'Episode,Airline,Price,Passengers,Revenue',
      ...results.flatMap(result => 
        Object.entries(result.airlines).map(([airline, data]: [string, any]) =>
          `${result.episode},${airline},${data.price},${data.passengers},${data.revenue}`
        )
      )
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'simulation_results.csv';
    a.click();
  };

  // Prepare chart data
  const chartData = results.length > 0 ? AIRLINES.map(airline => {
    const airlineResults = results.map(r => r.airlines[airline.code]);
    const avgRevenue = airlineResults.reduce((sum, r) => sum + r.revenue, 0) / airlineResults.length;
    const avgPrice = airlineResults.reduce((sum, r) => sum + r.price, 0) / airlineResults.length;
    const avgPassengers = airlineResults.reduce((sum, r) => sum + r.passengers, 0) / airlineResults.length;
    
    return {
      name: airline.code,
      revenue: avgRevenue,
      price: avgPrice,
      passengers: avgPassengers,
      color: airline.color
    };
  }) : [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Simulation</h1>
        <p className="text-gray-600 mt-2">
          Run competitive pricing simulations with trained PPO agents
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Simulation Controls */}
        <div className="lg:col-span-1">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">Simulation Controls</h3>
            
            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Episodes
                </label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={episodes}
                  onChange={(e) => setEpisodes(parseInt(e.target.value))}
                  className="input-field"
                  disabled={isRunning}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Mode
                </label>
                <select className="input-field" disabled={isRunning}>
                  <option>Deterministic</option>
                  <option>Stochastic</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleRunSimulation}
              disabled={isRunning}
              className="btn-primary w-full flex items-center justify-center space-x-2 mb-4"
            >
              <Play className="w-4 h-4" />
              <span>{isRunning ? 'Running...' : 'Run Simulation'}</span>
            </button>

            {results.length > 0 && (
              <button
                onClick={exportResults}
                className="btn-secondary w-full flex items-center justify-center space-x-2"
              >
                <Download className="w-4 h-4" />
                <span>Export CSV</span>
              </button>
            )}

            {isRunning && (
              <div className="mt-4">
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>Progress</span>
                  <span>{Math.floor(progress)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-3">
          {results.length > 0 ? (
            <div className="space-y-8">
              {/* Revenue Chart */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Average Performance</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
                      <YAxis stroke="#6b7280" fontSize={12} />
                      <Tooltip 
                        formatter={(value: number, name: string) => [
                          name === 'revenue' ? formatCurrency(value) : value.toFixed(0),
                          name.charAt(0).toUpperCase() + name.slice(1)
                        ]}
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                        }}
                      />
                      <Legend />
                      <Bar dataKey="revenue" fill="#3b82f6" name="Revenue" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Summary Statistics */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Summary Statistics</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Airline
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Avg Price
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Avg Passengers
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Avg Revenue
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Win Rate
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {chartData.map((airline) => {
                        const winRate = Math.random() * 0.4 + 0.1; // Mock win rate
                        return (
                          <tr key={airline.name} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <div 
                                  className="w-4 h-4 rounded-full mr-3"
                                  style={{ backgroundColor: airline.color }}
                                />
                                <span className="text-sm font-medium text-gray-900">
                                  {airline.name}
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {formatCurrency(airline.price)}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {airline.passengers.toFixed(0)}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {formatCurrency(airline.revenue)}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {(winRate * 100).toFixed(1)}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="text-center py-12">
                <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Simulation Results</h3>
                <p className="text-gray-500 mb-6">
                  Run a simulation to see competitive pricing results and agent performance.
                </p>
                <button
                  onClick={handleRunSimulation}
                  disabled={isRunning}
                  className="btn-primary"
                >
                  Run Your First Simulation
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};