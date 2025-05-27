// visualizationComponents.js
import Histogram from './Histogram';
import Line from './Line';
// import StockChart from "./StockChart";
import MultiStockChart from "./MultiStockChart";
import MultiLine from "./MultiLine";
import Editor from '../../../../Input/Editor';
import ChatScreen from './ChatScreen';
import PhotoDisplay from './PhotoDisplay';
import RecipeInstructions from './Recipe/RecipeInstructions';
import RecipeListItem from './Recipe/RecipeListItem';
import RecipeList from './Recipe/RecipeList';
import NewLine from './newLine';
import EditorList from './Editor/EditorList';
import EditorListItem from './Editor/EditorListItem';
import MealPlanView from './MealPlanner/MealPlanView';

const visualizationComponents = {
  histogram: Histogram,
  linegraph: Line,
  stockchart: MultiStockChart,
  multiline: MultiLine,
  editor: Editor,
  chat: ChatScreen,
  photo: PhotoDisplay,
  recipeinstructions: RecipeInstructions,
  recipelistitem: RecipeListItem,
  recipelist: RecipeList,
  newline: NewLine,
  editorlist: EditorList,
  document_summary_view: EditorListItem,
  mealplan: MealPlanView,
};

export default visualizationComponents;