// visualizationComponents.js
import Histogram from './Histogram';
import Line from './Line';
import StockChart from "./StockChart";
import MultiStockChart from "./MultiStockChart";
import MultiLine from "./MultiLine";
// Import other visualization components as needed

const visualizationComponents = {
  histogram: Histogram,
  linegraph: Line,
  stockchart: MultiStockChart,
  multiline: MultiLine
  // MultiStockChart: MultiStockChart,
  // Add other mappings here
};

export default visualizationComponents;