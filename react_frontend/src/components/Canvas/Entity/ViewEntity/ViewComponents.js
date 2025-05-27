import Histogram from './Histogram';
import Line from './Line';
import MultiStockChart from "./MultiStockChart";
import MultiLine from "./MultiLine";
import Editor from '../../../../Input/Editor';
import ChatView from './ChatView'; // Import the new ChatView component
import PhotoDisplay from './PhotoDisplay';

const viewComponents = {
  histogram: Histogram,
  linegraph: Line,
  stockchart: MultiStockChart,
  multiline: MultiLine,
  editor: Editor,
  chat: ChatView, // Use ChatView for the 'chat' key
  photo: PhotoDisplay
};

export default viewComponents;