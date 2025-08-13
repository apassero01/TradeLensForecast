// visualizationComponents.js
import Histogram from './Histogram';
import Line from './Line';
// import StockChart from "./StockChart";
import MultiStockChart from "./MultiStockChart";
import MultiLine from "./MultiLine";
import Editor from '../../../../Input/Editor';
import ChatInterface from './ChatInterface';
import PhotoDisplay from './PhotoDisplay';
import RecipeInstructions from './Recipe/RecipeInstructions';
import RecipeListItem from './Recipe/RecipeListItem';
import RecipeList from './Recipe/RecipeList';
import NewLine from './newLine';
import MealPlanView from './MealPlanner/MealPlanView';
import MealPlannerDashboard from './MealPlanner/MealPlannerDashboard';
import EntityRenderer from './EntityRenderer/EntityRenderer';
import IdeAppDashboard from './DocumentEditor/IdeAppDashboard';
import FileTree from './DocumentEditor/FileTree';
import DocumentSearch from './DocumentEditor/DocumentSearch';
import AdvancedDocumentEditor from './DocumentEditor/AdvancedDocumentEditor';
import DocumentListItem from './DocumentEditor/DocumentListItem';
import CalendarEventDetails from './CalendarEntity/CalendarEventDetails';
import CalendarMonthlyView from './CalendarEntity/CalendarMonthlyView';
import { EntityCentricChatView } from './EntityCentricChatView';
import MainUIPhase1 from './MainUIPhase1/MainUIPhase1';
import Phase1UIPrototype from './Phase1UIPrototype/Phase1UIPrototype';

const viewComponents = {
  histogram: { component: Histogram, displayName: 'Histogram' },
  linegraph: { component: Line, displayName: 'Line Graph' },
  stockchart: { component: MultiStockChart, displayName: 'Stock Chart' },
  multiline: { component: MultiLine, displayName: 'Multi Line' },
  editor: { component: Editor, displayName: 'Editor' },
  chatinterface: { component: ChatInterface, displayName: 'Chat Interface' },
  photo: { component: PhotoDisplay, displayName: 'Photo Display' },
  recipeinstructions: { component: RecipeInstructions, displayName: 'Recipe Instructions' },
  recipelistitem: { component: RecipeListItem, displayName: 'Recipe List Item' },
  recipelist: { component: RecipeList, displayName: 'Recipe List' },
  newline: { component: NewLine, displayName: 'New Line' },
  document_list_item: { component: DocumentListItem, displayName: 'Document List Item' },
  mealplan: { component: MealPlanView, displayName: 'Meal Plan' },
  mealplannerdashboard: { component: MealPlannerDashboard, displayName: 'Meal Planner Dashboard' },
  entityrenderer: { component: EntityRenderer, displayName: 'Entity Renderer' },
  ide_app_dashboard: { component: IdeAppDashboard, displayName: 'IDE App Dashboard' },
  file_tree: { component: FileTree, displayName: 'File Tree' },
  document_search: { component: DocumentSearch, displayName: 'Document Search' },
  advanced_document_editor: { component: AdvancedDocumentEditor, displayName: 'Advanced Document Editor' },
  calendar_event_details: { component: CalendarEventDetails, displayName: 'Calendar Event Details' },
  calendar_monthly_view: { component: CalendarMonthlyView, displayName: 'Calendar Monthly View' },
  entity_centric_chat_view: { component: EntityCentricChatView, displayName: 'Entity Centric Chat View' },
  main_ui_phase1: { component: MainUIPhase1, displayName: 'Main UI (Phase 1)' },
  phase1_ui_prototype: { component: Phase1UIPrototype, displayName: 'Phase 1 UI Prototype' },
};

// Helper function to get component names for backwards compatibility
export const getViewComponentNames = () => {
  const names = {};
  Object.keys(viewComponents).forEach(key => {
    names[key] = viewComponents[key].displayName;
  });
  return names;
};

// Helper function to get just the components for backwards compatibility
export const getViewComponents = () => {
  const components = {};
  Object.keys(viewComponents).forEach(key => {
    components[key] = viewComponents[key].component;
  });
  return components;
};

export default viewComponents;