import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

interface Ingredient {
  quantity: number;
  unit: string;
}

interface IngredientEntry {
  [ingredientName: string]: Ingredient;
}

interface RecipeData {
  /** free-form Markdown or a single-line numeric string like "1. Step one. 2. Step two." */
  instructions?: string;
  ingredients?: IngredientEntry[];
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

  const ingredientsArray = data?.ingredients;
  const hasIngredients = ingredientsArray && ingredientsArray.length > 0;

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-2xl shadow-lg nowheel overflow-auto">
      {hasIngredients && (
        <div className="mb-6">
          <h2 className="text-2xl font-semibold mb-4 text-gray-800">
            Ingredients
          </h2>
          <ul className="list-disc list-inside space-y-1">
            {ingredientsArray.map((ingredientObj, index) => {
              const ingredientName = Object.keys(ingredientObj)[0];
              const details = ingredientObj[ingredientName];
              return (
                <li key={`${ingredientName}-${index}`} className="text-gray-700">
                  {ingredientName}: {details.quantity} {details.unit}
                </li>
              );
            })}
          </ul>
        </div>
      )}

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