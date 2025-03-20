from shared_utils.strategy.BaseStrategy import Strategy, CreateEntityStrategy, HTTPGetRequestStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List
from dataclasses import dataclass
import json
import re

@dataclass
class Message:
    type: str
    content: str

    def serialize(self):
        return {
            'type': self.type,
            'content': self.content
        }


class ConfigureApiModelStrategy(Strategy):
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Configures an API model with connection details and parameters'

    def verify_executable(self, entity, strategy_request):
        config = strategy_request.param_config
        required = ['model_type', 'env_key']
        return all(k in config for k in required)

    def apply(self, entity) -> StrategyRequestEntity:
        config = self.strategy_request.param_config
        
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv(config['env_key'])
        if not api_key:
            raise ValueError(f"API key not found in environment for key: {config['env_key']}")

        # Set model type
        model_type = config.get('model_type', 'openai')
        entity.set_attribute('model_type', model_type)
        
        # Set model name with defaults based on type
        model_name = config.get('model_name')
        if not model_name:
            if model_type == 'openai':
                model_name = 'gpt-4o-mini'
        entity.set_attribute('model_name', model_name)
        
        # Set API key
        entity.set_attribute('api_key', api_key)
        
        # Set config with OpenAI defaults if not provided
        default_config = {
            'temperature': 0.7,
            'max_tokens': 5000,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'stream': False
        }
        
        if 'model_config' in config:
            default_config.update(config['model_config'])
        
        entity.set_attribute('config', default_config)
            
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'model_type': 'openai',  # Optional, defaults to openai
            'model_name': 'gpt-4o-mini',  # Optional, has defaults per model_type
            'env_key': 'OPENAI_API_KEY',  # Required - environment variable key
            'model_config': {  # Optional - has defaults
                'temperature': 0.7,
                'max_tokens': 4000,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0,
                'stream': False
            }
        }
    

class CallApiModelStrategy(Strategy):
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Makes a call to the configured API model using LangChain'

    def verify_executable(self, entity, strategy_request):
        required_attrs = ['model_type', 'model_name', 'api_key']
        return all(entity.has_attribute(attr) for attr in required_attrs)

    def form_context(self, entity) -> str:
        """Form context from document children"""
        doc_ids = self.entity_service.get_children_ids_by_type(entity, EntityEnum.DOCUMENT)

        history = entity.get_attribute('message_history')
        
        if not doc_ids:
            return ""
            
        contexts = []
        for doc_id in doc_ids:
            doc = self.entity_service.get_entity(doc_id)
            if doc and doc.has_attribute('text'):
                doc_type = doc.get_document_type() or 'unknown'
                if doc.has_attribute('path'):
                    doc_type = f"{doc_type} ({doc.get_attribute('path')})"
                if doc.has_attribute('name'):
                    doc_type = f"{doc_type} - {doc.get_attribute('name')}"
                contexts.append(
                    f"{'='*50}\n"
                    f"Document Type and Name and Path : {doc_type.upper()}\n"
                    f"{'-'*50,'DOCUMENT_BEGIN'}\n"
                    f"{doc.get_text()}\n"
                    f"{'='*50,"DOCUMENT_END"}\n"
                )

        strategy_directory = self.get_strategy_directory(entity)
        contexts.append(
            f"{'='*50}\n"
            f"Strategy Directory\n"
            f"{'-'*50}\n"
            f"{strategy_directory}\n"
            f"{'='*50}\n"
        )

        available_entities = self.get_available_entities(entity)
        contexts.append(
            f"{'='*50}\n"
            f"Available Entities\n"
            f"{'-'*50}\n"
            f"{available_entities}\n"
            f"{'='*50}\n"
        )

        entity_graph = self.serialize_entity_and_children(entity.entity_id)
        contexts.append(
            f"{'='*50}\n"
            f"Entity Graph\n"
            f"{'-'*50}\n"
            f"{json.dumps(entity_graph, indent=2)}\n"
            f"{'='*50}\n"
        )

        for message in history:
            contexts.append(
                f"{'='*50}\n"
                f"Message Type: {message.type.upper()}\n"
                f"{'-'*50}\n"
                f"{message.content}\n"
                f"{'='*50}\n"
            )
        
        return "\n".join(contexts)

    def format_response_text(self, text: str, max_line_length: int = 80) -> str:
        """Format response text with proper line breaks"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if adding this word exceeds max line length
            if current_length + len(word) + 1 <= max_line_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                # Start new line
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        # Add final line
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines)

    def apply(self, entity) -> StrategyRequestEntity:
        # Get context from documents
        context = self.form_context(entity)
        
        # Get user input and system prompt from config
        config = self.strategy_request.param_config

        if entity.has_attribute('user_input'):
            user_input = entity.get_attribute('user_input')
        else:
            user_input = config.get('user_input', '')

        system_prompt = config.get('system_prompt', '')
        context_prefix = config.get('context_prefix', 'Here is the relevant context:')
        
        # Combine context with user input
        if context:
            combined_input = (
                f"{context_prefix}\n\n"
                f"{context}\n\n"
                f"{'='*50}\n"
                f"USER QUERY:\n"
                f"{'-'*50}\n"
                f"{user_input}\n"
                f"{'='*50}"
            )
        else:
            combined_input = user_input

        # Initialize LangChain chat model
        chat = ChatOpenAI(
            model_name=entity.get_attribute('model_name'),
            openai_api_key=entity.get_attribute('api_key'),
            temperature=entity.get_attribute('config').get('temperature', 0.7),
            max_tokens=entity.get_attribute('config').get('max_tokens', 1000)
        )

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=combined_input))

        # Use invoke() instead of __call__
        response = chat.invoke(messages)
        
        # Get message history
        history = entity.get_attribute('message_history')
        
        # Format complete response with history
        
        # Update history and attributes
        history.append(Message(type='context', content=user_input))
        history.append(Message(type='response', content=response.content))

        self.parse_strategy_requests(response.content, entity)
        
        entity.set_attribute('last_context', context)
        entity.set_attribute('message_history', history)
        entity.set_attribute('response', [message.serialize() for message in history])
        
        self.strategy_request.ret_val['response'] = history
        self.strategy_request.ret_val['context_used'] = context
        
        return self.strategy_request

    def parse_strategy_requests(self, message: str, entity: 'Entity'):
        """
        Parse strategy request blocks from a given message.

        This function looks for blocks marked with:

            ```Strategy
            { JSON CODE }
            ```

        It then converts the JSON code inside these blocks into Python dictionaries
        and returns them in a list called strategy_request_blob.

        :param message: A string containing the message with potential strategy blocks.
        :return: A list of dictionaries representing the parsed JSON strategy requests.
        """
        # Regex to find blocks starting with ```Strategy and ending with ```
        pattern = r"```StrategyRequest\s*\n([\s\S]*?)\n```"
        matches = re.findall(pattern, message, re.MULTILINE)

        strategy_requests = []

        for block in matches:
            try:
                # Strip whitespace and convert the JSON code to a Python dict
                parsed_json = json.loads(block.strip())

                child_request = StrategyRequestEntity()
                child_request.strategy_name = CreateEntityStrategy.__name__
                child_request.target_entity_id = entity.entity_id
                child_request.param_config = {
                    'entity_class': "shared_utils.entities.StrategyRequestEntity.StrategyRequestEntity",
                    'entity_uuid': None
                }
                child_request = self.executor_service.execute_request(child_request)
                request = child_request.ret_val['child_entity']
                request.strategy_name = parsed_json['strategy_name']
                request.param_config = parsed_json['param_config']
                request.add_to_history = parsed_json['add_to_history']
                request.target_entity_id = parsed_json['target_entity_id']

                self.entity_service.save_entity(request)
                entity.add_child(request)

            except Exception as e:
                print(f"Error parsing JSON in strategy block: {e}")

    def get_strategy_directory(self, entity: 'Entity'):
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = HTTPGetRequestStrategy.__name__
        strategy_request.target_entity_id = entity.entity_id
        strategy_request.param_config = {
            'url': 'http://127.0.0.1:8000/training_session/api/get_strategy_registry/',
            'response_attribute': 'strategy_registry'
        }
        strategy_request = self.executor_service.execute_request(strategy_request)
        return strategy_request.ret_val['strategy_registry']

    def get_available_entities(self, entity: 'Entity'):
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = HTTPGetRequestStrategy.__name__
        strategy_request.target_entity_id = entity.entity_id
        strategy_request.param_config = {
            'url': 'http://127.0.0.1:8000/training_session/api/get_available_entities/',
            'response_attribute': 'available_entities'
        }
        strategy_request = self.executor_service.execute_request(strategy_request)
        return strategy_request.ret_val['available_entities']

    def serialize_entity_and_children(self, entity_id, return_dict=None):
        try:
            entity = self.entity_service.get_entity(entity_id)
        except Exception as e:
            return return_dict
        if not entity:
            return None
        entity_dict = entity.serialize()
        children = entity.get_children()
        if return_dict is None:
            return_dict = {}

        if entity_id in return_dict:
            return return_dict
        return_dict[entity_id] = entity_dict

        for child_id in children:
            return_dict = self.serialize_entity_and_children(child_id, return_dict)

        return return_dict

    @staticmethod
    def get_request_config():
        return {
            'user_input': '',  # Optional additional input
            'system_prompt': '',  # Optional system prompt
            'context_prefix': 'Here is the relevant context:',  # Optional prefix for context
        }
    
