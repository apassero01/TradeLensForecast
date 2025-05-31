# ChatInterface Component

A modern, feature-rich chat interface component that follows the MealPlanner architectural patterns and provides seamless integration with AI models through strategy requests.

## Features

- **Full Strategy Integration**: Uses `CallApiModelStrategy` and `ConfigureApiModelStrategy` for AI interactions
- **Automatic Model Management**: Creates and configures API models automatically
- **Modern UI**: Clean, responsive design with dark theme
- **Rich Message Display**: Supports markdown, code blocks, and syntax highlighting
- **Configurable Settings**: System prompts, display modes, font sizing
- **Real-time Status**: Shows model status and message counts
- **Copy Functionality**: Easy copying of messages and code blocks
- **Auto-scroll**: Intelligent scrolling behavior

## Usage

### As a View Component

Use `chatinterface` as the `view_component_type` when creating a view:

```javascript
sendStrategyRequest(StrategyRequests.createEntity(
  parentEntityId,
  'view',
  {
    view_component_type: 'chatinterface',
    name: 'AI Assistant',
    system_prompt: 'You are a helpful AI assistant.',
    auto_scroll: true,
    show_settings: false
  }
));
```

### Data Structure

The component expects data in this format:

```typescript
interface ChatInterfaceData {
  name?: string;                    // Display name for the chat
  system_prompt?: string;           // Default system prompt
  model_type?: string;              // AI model type (default: 'openai')
  model_name?: string;              // AI model name (default: 'gpt-4o-mini')
  message_history?: Message[];      // Chat message history
  current_input?: string;           // Current input text
  auto_scroll?: boolean;            // Auto-scroll to bottom
  show_settings?: boolean;          // Show settings panel
}

interface Message {
  type: 'request' | 'response' | 'context';
  content: string;
  timestamp?: string;
}
```

## Strategy Requests

The component automatically handles these strategy requests:

### Model Creation
- **CreateEntityStrategy**: Creates a new API model entity when none exists
- **ConfigureApiModelStrategy**: Configures the API model with OpenAI settings

### Chat Operations
- **CallApiModelStrategy**: Sends messages to the AI model
- **ClearChatHistoryStrategy**: Clears the chat history

### Data Persistence
- **SetAttributesStrategy**: Saves view state and settings

## Key Differences from Original ChatScreen

1. **Interactive Input**: Full input interface with submit functionality
2. **Model Management**: Automatic API model creation and configuration
3. **Strategy Integration**: Uses proper strategy request patterns
4. **Modern Design**: Updated UI with better UX
5. **Settings Panel**: Configurable system prompts and display options
6. **Status Indicators**: Real-time model and submission status

## Architecture

The component follows the established patterns from MealPlanner:

- Uses `useRecoilValue` with `childrenByTypeSelector` to find API models
- Implements proper strategy request building with `StrategyRequests.builder()`
- Follows the component interface pattern with `sendStrategyRequest` and `updateEntity`
- Maintains separation of concerns between UI state and entity data

## Example Implementation

```javascript
// In a parent component or strategy
const createChatView = () => {
  sendStrategyRequest(StrategyRequests.builder()
    .withStrategyName('CreateEntityStrategy')
    .withTargetEntity(parentEntityId)
    .withParams({
      entity_class: 'view',
      initial_attributes: {
        view_component_type: 'chatinterface',
        name: 'AI Research Assistant',
        system_prompt: 'You are a research assistant specialized in data analysis.',
        show_settings: true
      }
    })
    .build());
};
```

The ChatInterface will then:
1. Automatically find or create an API model
2. Configure it with OpenAI settings
3. Handle user input and AI responses
4. Maintain conversation history
5. Provide rich message display with code highlighting 