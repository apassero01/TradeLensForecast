// visualizationComponents.js
import Histogram from './Histogram';
import Line from './Line';
// import StockChart from "./StockChart";
import MultiStockChart from "./MultiStockChart";
import MultiLine from "./MultiLine";
import Editor from '../../../../Input/Editor';
import ChatScreen from './ChatScreen';
import ChatInterface from './ChatInterface';
import PhotoDisplay from './PhotoDisplay';
import RecipeInstructions from './Recipe/RecipeInstructions';
import RecipeListItem from './Recipe/RecipeListItem';
import RecipeList from './Recipe/RecipeList';
import NewLine from './newLine';
import EditorList from './Editor/EditorList';
import MealPlanView from './MealPlanner/MealPlanView';
import MealPlannerDashboard from './MealPlanner/MealPlannerDashboard';
import EntityRenderer from './EntityRenderer/EntityRenderer';
import IdeAppDashboard from './DocumentEditor/IdeAppDashboard';
import FileTree from './DocumentEditor/FileTree';
import DocumentSearch from './DocumentEditor/DocumentSearch';
import AdvancedDocumentEditor from './DocumentEditor/AdvancedDocumentEditor';
import DocumentListItem from './DocumentEditor/DocumentListItem';

const visualizationComponents = {
  histogram: Histogram,
  linegraph: Line,
  stockchart: MultiStockChart,
  multiline: MultiLine,
  editor: Editor,
  chat: ChatScreen,
  chatinterface: ChatInterface,
  photo: PhotoDisplay,
  recipeinstructions: RecipeInstructions,
  recipelistitem: RecipeListItem,
  recipelist: RecipeList,
  newline: NewLine,
  editorlist: EditorList,
  document_summary_view: DocumentListItem,
  document_list_item: DocumentListItem,
  mealplan: MealPlanView,
  mealplannerdashboard: MealPlannerDashboard,
  entityrenderer: EntityRenderer,
  ide_app_dashboard: IdeAppDashboard,
  file_tree: FileTree,
  document_search: DocumentSearch,
  advanced_document_editor: AdvancedDocumentEditor,
};

export default visualizationComponents;