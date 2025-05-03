import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

interface RecipeData {
  /** free-form Markdown or a single-line numeric string like "1. Step one. 2. Step two." */
  instructions?: string;
}

interface RecipeInstructionsProps {
  data?: RecipeData;
}

export default function RecipeInstructions({ data }: RecipeInstructionsProps) {
  // 1) grab the raw string or fallback
  const raw = data?.instructions?.trim() || '';

  // 2) if it's all one line with embedded numbers, split into lines:
  //    "1. A 2. B 3. C" → "\n1. A\n2. B\n3. C\n"
  const withLines =
    raw && !raw.includes('\n') && /\d\.\s*/.test(raw)
      ? raw
          .replace(/\s*(\d+)\.\s*/g, '\n$1. ')
          .trim() + '\n'
      : raw;

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-2xl shadow-lg nowheel overflow-auto">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800 overflow-auto">
        Instructions
      </h2>

      {withLines ? (
        <article className="prose prose-lg prose-gray overflow-auto">
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
            {withLines}
          </ReactMarkdown>
        </article>
      ) : (
        <p className="text-gray-500 italic">No instructions available.</p>
      )}
    </div>
  );
}