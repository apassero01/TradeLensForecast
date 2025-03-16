// visualizationComponents.js
import Histogram from './Histogram';
import Line from './Line';
// import StockChart from "./StockChart";
import MultiStockChart from "./MultiStockChart";
import MultiLine from "./MultiLine";
import Editor from '../../../../Input/Editor';
import ChatScreen from './ChatScreen';
import PhotoDisplay from './PhotoDisplay';

const visualizationComponents = {
  histogram: Histogram,
  linegraph: Line,
  stockchart: MultiStockChart,
  multiline: MultiLine,
  editor: Editor,
  chat: ChatScreen,
  photo: PhotoDisplay
};

export default visualizationComponents;