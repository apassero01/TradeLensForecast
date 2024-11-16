// visualizationComponents.js
import Histogram from './Histogram';
import MultiHistogram from "./MultiHistogram";
import Line from "./Line";
import MultiLine from "./MultiLine";
import SequenceMultiLine from "./SequenceMultiLine";
// Import other visualization components as needed

const visualizationComponents = {
  histogram: Histogram,
  multihist : MultiHistogram,
  line: Line,
  multiline: MultiLine,
  sequence_multiline: SequenceMultiLine
  // Add other mappings here
};

export default visualizationComponents;