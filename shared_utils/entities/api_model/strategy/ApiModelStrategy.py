import datetime
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from shared_utils.strategy.BaseStrategy import Strategy, CreateEntityStrategy, HTTPGetRequestStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
from typing import List
from dataclasses import dataclass
import json
import re
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# @dataclass
# class Message:
#     type: str
#     content: str

#     def serialize(self):
#         return {
#             'type': self.type,
#             'content': self.content
#         }


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

        # Set model type
        model_type = config.get('model_type', 'openai')
        entity.set_attribute('model_type', model_type)
        
        # Set model name with defaults based on type
        model_name = config.get('model_name')
        if not model_name:
            if model_type == 'openai':
                model_name = 'gpt-4o-mini'
        entity.set_attribute('model_name', model_name)

        if model_type == 'openai':
            env_var = 'OPENAI_API_KEY'
        elif model_type == 'anthropic':
            env_var = 'ANTHROPIC_API_KEY'
        elif model_type == 'google_genai':
            env_var = 'GOOGLE_API_KEY'
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        api_key = os.getenv(env_var)
        
        # Set API key
        entity.set_attribute('api_key', api_key)
        
        # Set config with OpenAI defaults if not provided
        default_config = {
            'max_tokens': 5000,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'stream': False
        }
        
        if 'model_config' in config:
            default_config.update(config['model_config'])
        
        entity.set_attribute('config', default_config)

        self.entity_service.save_entity(entity)
            
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'model_type': 'openai',  # Optional, defaults to openai
            'model_name': 'gpt-4o-mini',  # Optional, has defaults per model_type
            'env_key': 'OPENAI_API_KEY',  # Required - environment variable key
            'model_config': {  # Optional - has defaults
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
        required_attrs = ['model_type', 'model_name', 'api_key', 'serialize_entities_and_strategies']
        return all(entity.has_attribute(attr) for attr in required_attrs)

    def form_context(self, entity) -> str:
        """Form context from document children"""
        doc_ids = self.entity_service.get_children_ids_by_type(entity, EntityEnum.DOCUMENT)

        if not doc_ids:
            return ""

        contexts = []

        instructions_doc_id = "c226b236-5567-49fe-98dc-26c65d50397a"
        instructions_doc = self.entity_service.get_entity(instructions_doc_id)
        if instructions_doc:
            contexts.append(
                f"{'='*50}\n"
                f"SYSTEM INSTRUCTIONS DO NOT DEVIATE FROM THESE INSTRUCTIONS\n"
                f"{'-'*50}\n"
                f"{instructions_doc.get_attribute('text')}\n"
                f"{'='*50}\n"
            )
        
        if instructions_doc_id in doc_ids:
            doc_ids.remove(instructions_doc_id)

        contexts.append("These documents may contain import instructions or other relevant information:")
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
                    f"Document Type and Name and Path : {doc_type.upper() + " DocumentID: "  + doc_id}\n"
                    f"{'-'*50,'DOCUMENT_BEGIN'}\n"
                    f"{doc.get_text()}\n"
                    f"{'='*50,"DOCUMENT_END"}\n"
                )

        if self.strategy_request.param_config.get('serialize_entities_and_strategies', False):
            entity_graph = self.serialize_entity_and_children(entity.entity_id)
            contexts.append(
                f"{'=' * 50}\n"
                f"Entity Graph\n"
                f"{'-' * 50}\n"
                f"{json.dumps(entity_graph, indent=2)}\n"
                f"{'=' * 50}\n"
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

        contexts.append("HERE IS THE CURRENT DATE USE IT IF THE USER REQUESTS DATE RELEVANT INFORMATION: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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

        user_input = config.get('user_input', '')

        system_prompt = config.get('system_prompt', '')
        context_prefix = config.get('context_prefix', 'Here is the relevant context:')


        # # Initialize LangChain chat model
        # chat = ChatOpenAI(
        #     model_name=entity.get_attribute('model_name'),
        #     openai_api_key=entity.get_attribute('api_key'),
        #     max_tokens=entity.get_attribute('config').get('max_tokens', 1000)
        # )
        chat = init_chat_model(
            model=entity.get_attribute('model_name'), 
            model_provider=entity.get_attribute('model_type'), 
            api_key=entity.get_attribute('api_key')
        )
        chat = chat.bind_tools([self.create_strategy_request, self.yield_to_user])
        tool_dict = {"create_strategy_request": self.create_strategy_request, "yield_to_user": self.yield_to_user}

        context_list = []

        context_list.append(SystemMessage(content=context))

        MAX_ITERS = 25
        CUR_ITERS = 0
        # Use invoke() instead of __call__

        self.add_to_message_history(entity, HumanMessage(content=user_input))
        while CUR_ITERS < MAX_ITERS:
            CUR_ITERS += 1
            model_input = context_list + [SystemMessage(content="HERE IS THE CURRENT CONVERSATION HISTORY: ")] + entity.get_attribute('message_history')
            response = chat.invoke(model_input)
            if response.content != '':
                self.add_to_message_history(entity, AIMessage(content=response.content))
            if response.content == '' and response.tool_calls == []:
                self.add_to_message_history(entity, AIMessage(content="No response from model.... please call yield_to_user() tool to end the conversation."))

            self.entity_service.save_entity(entity)
            for tool_call in response.tool_calls:
                if tool_call['name'].lower() == 'yield_to_user':
                    CUR_ITERS = MAX_ITERS
                    break
                selected_tool = tool_dict.get(tool_call['name'].lower())
                tool_msg = selected_tool.invoke(tool_call)
                child_request = tool_msg.artifact
                try:
                    request = self.execute_model_request(child_request, entity)
                    entity = self.entity_service.get_entity(entity.entity_id)
                    ret_val = request.ret_val
                    if 'entity' in ret_val:
                        del ret_val['entity']
                    target_entity = self.entity_service.get_entity(request.target_entity_id)
                    request_message = SystemMessage(content=f"Result of Model Executed strategy {request.strategy_name} with config {request.param_config} on target entity {request.target_entity_id} This step is complete: Are there any further actions needed? If yes, complete further actions, else let the user return additional information be sure to call the yield_to_user() tool")

                    if target_entity.entity_id != entity.entity_id:
                        updated_entity_message = SystemMessage(content=self.format_entity_response(target_entity.serialize()))
                        request_message.content += f"\n\n{updated_entity_message.content}"

                    self.add_to_message_history(entity, request_message)
                except Exception as e:
                    error_message = f"Error executing tool {tool_call['name']}: {str(e)}"
                    logger.error(error_message)
                    self.add_to_message_history(entity, SystemMessage(content=error_message))
                    continue




        logger.info(f"Response from model: Message Received from APi model")
        
        self.entity_service.save_entity(entity)
        
        return self.strategy_request

    def execute_model_request(self, request, entity):
        target_entity = self.entity_service.get_entity(request.target_entity_id)
        if target_entity is None:
            raise ValueError(f'Entity not found for id: {request.target_entity_id}')
        target_entity.add_child(request)
        self.entity_service.save_entity(target_entity)
        self.entity_service.save_entity(request)
        request = self.executor_service.execute_request(request)

        return request


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
        children = entity.get_children()
        if return_dict is None:
            return_dict = {}

        if entity_id in return_dict:
            return return_dict
        return_dict[entity_id] = {
            'entity_id' : entity_id,
            'entity_type': entity.entity_name.value,
            'children_ids': children,
            'name' : entity.get_attribute('name') if entity.has_attribute('name') else None,
        }

        for child_id in children:
            return_dict = self.serialize_entity_and_children(child_id, return_dict)

        return return_dict

    @tool()
    def serialize_entities(self,entities: List[str]) -> List[dict]:
        '''
        Serialize a list of entities. If the model needs to know about entities with
        specific ids, this method will return more information about the entities.
        '''
        serialized_entities = []
        for entity_id in entities:
            entity = self.entity_service.get_entity(entity_id)
            if entity:
                serialized_entities.append(entity.serialize())
        return serialized_entities

    @staticmethod
    @tool
    def yield_to_user() -> str:
        '''
        This tool is to yield control of the conversation back to the user when input is needed before further work.
        '''
        return ''

    @staticmethod
    @tool(response_format="content_and_artifact")
    def create_strategy_request(strategy_name: str, param_config: dict | str, target_entity_id: str, add_to_history: bool = False, target_entity_ids: List[str] = None) -> str:
        '''
        Create a strategy request entity with the given parameters.
        @param strategy_name: The name of the strategy to execute
        @param MUST BE JSON param_config: The configuration parameters for the strategy
        @param target_entity_id: The id of the target entity for the strategy
        @param add_to_history: Whether to add the strategy request to the entity's history
        @param target_entity_ids: List of target entity ids for the strategy if we are executing the request on multiple entities
        '''
        if isinstance(param_config, str):
            param_config = json.loads(param_config)


        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = strategy_name
        strategy_request.param_config = param_config
        strategy_request.target_entity_id = target_entity_id
        strategy_request.add_to_history = False
        strategy_request.set_attribute('target_entity_ids', target_entity_ids)
        return "Created Strategy Request", strategy_request

    def add_to_message_history(self, entity, message):
        """
        Add a message to the message history of the entity.
        """
        history = entity.get_attribute('message_history')
        history.append(message)
        self.entity_service.save_entity(entity)


    @staticmethod
    def get_request_config():
        return {
            'user_input': '',  # Optional additional input
            'system_prompt': '',  # Optional system prompt
            'context_prefix': 'Here is the relevant context:',  # Optional prefix for context\
            'serialize_entities_and_strategies': False # Optional flag to serialize entities and strategies
        }

    def format_entity_response(self, entities_dict):
        entity_graph_json = json.dumps(entities_dict, indent=2)

        response = f"""
    
        {'=' * 50}
        Entity Graph
        {'-' * 50}
        {entity_graph_json}
        {'=' * 50}
        """
        return response


class ClearChatHistoryStrategy(Strategy):
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Clears the chat history for an API model'

    def apply(self, entity) -> StrategyRequestEntity:
        entity.set_attribute('message_history', [])
        entity.set_attribute('response', [])
        self.entity_service.save_entity(entity)
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {}
    
