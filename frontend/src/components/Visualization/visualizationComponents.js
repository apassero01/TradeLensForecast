// visualizationComponents.js
import Histogram from './Histogram';
import Line from './Line';
import StockChart from "./StockChart";
import MultiStockChart from "./MultiStockChart";
import MultiLine from "./MultiLine";
import Editor from './Editor';
import ChatScreen from './ChatScreen';
const visualizationComponents = {
  histogram: Histogram,
  linegraph: Line,
  stockchart: MultiStockChart,
  multiline: MultiLine,
  editor: Editor,
  chat: ChatScreen
};

export default visualizationComponents;