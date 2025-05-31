import React, { useState } from 'react';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { IoSearch, IoDocumentText, IoFolder } from 'react-icons/io5';

interface DocumentSearchProps {
  data?: DocumentSearchData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
}

interface DocumentSearchData {
  // No specific data needed from parent
}

interface SearchResult {
  entity_id: string;
  name: string;
  docName: string;
  is_folder: boolean;
  file_type: string;
  path: string;
  match_type: string;
  snippet?: string;
}

export default function DocumentSearch({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}: DocumentSearchProps) {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState<'name' | 'docName' | 'content'>('name');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setIsSearching(true);
    setHasSearched(true);

    const searchRequest = StrategyRequests.builder()
      .withStrategyName('SearchDocumentsStrategy')
      .withTargetEntity(parentEntityId)
      .withParams({
        query: query.trim(),
        search_type: searchType
      })
      .withAddToHistory(false)
      .build();

    // Send the request - results will come back through entity updates
    sendStrategyRequest(searchRequest);
    
    // Clear the searching state after a delay
    setTimeout(() => {
      setIsSearching(false);
    }, 1000);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleResultClick = (entityId: string) => {
    // TODO: Notify parent to open this document
    console.log('Open document:', entityId);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Search Input */}
      <div className="flex flex-col gap-2">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search..."
            className="flex-1 px-3 py-1 bg-gray-800 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={handleSearch}
            disabled={!query.trim() || isSearching}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-sm flex items-center gap-1"
          >
            {/* @ts-ignore */}
            <IoSearch size={14} />
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Search Type Selector */}
        <div className="flex gap-2 text-xs">
          <label className="flex items-center gap-1">
            <input
              type="radio"
              value="name"
              checked={searchType === 'name'}
              onChange={(e) => setSearchType('name')}
              className="text-blue-600"
            />
            <span>Name</span>
          </label>
          <label className="flex items-center gap-1">
            <input
              type="radio"
              value="docName"
              checked={searchType === 'docName'}
              onChange={(e) => setSearchType('docName')}
              className="text-blue-600"
            />
            <span>Display Name</span>
          </label>
          <label className="flex items-center gap-1">
            <input
              type="radio"
              value="content"
              checked={searchType === 'content'}
              onChange={(e) => setSearchType('content')}
              className="text-blue-600"
            />
            <span>Content</span>
          </label>
        </div>
      </div>

      {/* Results */}
      <div className="mt-3 flex-1 overflow-y-auto">
        {!hasSearched ? (
          <div className="text-center text-gray-500 text-xs mt-4">
            <p>Enter a search term above</p>
          </div>
        ) : isSearching ? (
          <div className="text-center text-gray-500 text-xs mt-4">
            <p>Searching...</p>
          </div>
        ) : results.length === 0 ? (
          <div className="text-center text-gray-500 text-xs mt-4">
            <p>No results found</p>
          </div>
        ) : (
          <div className="space-y-1">
            {results.map((result) => (
              <div
                key={result.entity_id}
                onClick={() => handleResultClick(result.entity_id)}
                className="p-2 hover:bg-gray-700 cursor-pointer rounded border border-transparent hover:border-gray-600"
              >
                <div className="flex items-start gap-2">
                  <span className="mt-0.5">
                    {result.is_folder ? (
                      /* @ts-ignore */
                      <IoFolder className="text-blue-400" size={14} />
                    ) : (
                      /* @ts-ignore */
                      <IoDocumentText className="text-gray-400" size={14} />
                    )}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">
                      {result.docName || result.name}
                      {result.docName && (
                        <span className="text-gray-500 ml-1 text-xs">
                          ({result.name})
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 truncate">
                      {result.path}
                    </div>
                    {result.snippet && (
                      <div className="text-xs text-gray-400 mt-1 line-clamp-2">
                        {result.snippet}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 