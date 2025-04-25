import Histogram from './Histogram';
import Line from './Line';
import MultiStockChart from "./MultiStockChart";
import MultiLine from "./MultiLine";
import Editor from '../../../../Input/Editor';
import ChatScreen from './ChatScreen';
import PhotoDisplay from './PhotoDisplay';

const viewComponents = {
  histogram: Histogram,
  linegraph: Line,
  stockchart: MultiStockChart,
  multiline: MultiLine,
  editor: Editor,
  chat: ChatScreen,
  photo: PhotoDisplay
};

export default viewComponents;