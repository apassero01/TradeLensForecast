import asyncio
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity

entity_service = EntityService()
strategy_executor_service = StrategyExecutorService()

class GlobalEntityConsumer(AsyncWebsocketConsumer):
    """Handles global session state and entity updates"""
    
    async def connect(self):
        print("Connecting to Global WebSocket")
        try:
            await self.accept()
            print("Connection accepted")
            
            await self.channel_layer.group_add("global_entities", self.channel_name)
            print("Added to global_entities group")
            
            await self.send(json.dumps({
                'type': 'connected',
                'message': 'Connected to global entity updates',
                'request_subscriptions': True  # Tell frontend to send subscriptions
            }))
            print("Sent connection confirmation")
            
        except Exception as e:
            print(f"Error in connect: {str(e)}")
            raise

    async def disconnect(self, close_code):
        print(f"Disconnecting from Global WebSocket with code: {close_code}")
        try:
            await self.channel_layer.group_discard("global_entities", self.channel_name)
            print("Removed from global_entities group")
        except Exception as e:
            print(f"Error in disconnect: {str(e)}")

    async def receive(self, text_data):
        """Handle incoming messages from the client"""
        try:
            data = json.loads(text_data)
            command = data.get('command')
            print(f"Received WebSocket command: {command}")
            print(f"Data: {data}")

            if command == 'subscribe_entities':
                entity_ids = data.get('entity_ids', [])
                await self.handle_entity_subscriptions(entity_ids)
            elif command == 'execute_strategy':
                await self.handle_execute_strategy(data.get('strategy'))
            elif command == 'start_session':
                await self.handle_start_session()
            elif command == 'stop_session':
                await self.handle_stop_session()
            elif command == 'delete_session':
                await self.handle_delete_session()
            elif command == 'ping':
                await self.send(json.dumps({
                    'type': 'pong',
                    'message': 'Pong'
                }))
            else:
                print(f"Unknown command received: {command}")
                await self.send(json.dumps({
                    'type': 'error',
                    'message': f'Unknown command: {command}'
                }))
        except Exception as e:
            print(f"Error in receive: {str(e)}")
            await self.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def handle_delete_session(self):
        """Handle deletion of the current session"""
        try:
            await sync_to_async(entity_service.delete_session)()
            await self.send(json.dumps({
                'type': 'session_deleted',
                'message': 'Session deleted'
            }))
        except Exception as e:
            print(f"Error deleting session: {str(e)}")
            await self.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    def json_to_strategy_request(self, json_data):
        if 'entity_id' in json_data:
            try:
                strat_request = entity_service.get_entity(json_data['entity_id'])
            except Exception as e:
                print(f"Error getting entity {json_data['entity_id']}: {str(e)}")
                strat_request = StrategyRequestEntity(json_data['entity_id'])
        else:
            strat_request = StrategyRequestEntity()

        strat_request.strategy_name = json_data['strategy_name']
        strat_request.param_config = json_data['param_config']
        strat_request.add_to_history = json_data['add_to_history']
        strat_request.target_entity_id = json_data['target_entity_id']

        nested_requests = json_data['nested_requests']
        for nested_request in nested_requests:
            strat_request.add_nested_request(self.json_to_strategy_request(nested_request))

        return strat_request

    async def handle_entity_subscriptions(self, entity_ids):
        """Handle initial entity subscriptions"""
        try:
            print(f"Setting up subscriptions for entities: {entity_ids}")
            
            # Get current entities from service
            entities_data = {}
            for entity_id in entity_ids:
                entity = await sync_to_async(entity_service.get_entity)(entity_id)
                if entity:
                    entities_data[entity_id] = entity.serialize()

            # Send current state of subscribed entities
            if entities_data:
                await self.send(json.dumps({
                    'type': 'entity_update',
                    'entities': entities_data
                }))

            await self.send(json.dumps({
                'type': 'subscriptions_complete',
                'message': 'Entity subscriptions processed'
            }))

        except Exception as e:
            print(f"Error setting up subscriptions: {str(e)}")
            await self.send(json.dumps({
                'type': 'error',
                'message': f'Failed to set up subscriptions: {str(e)}'
            }))

    async def handle_execute_strategy(self, strategy_data):
        """Handle strategy execution via WebSocket"""
        try:
            print(f"Executing strategy: {strategy_data}")
            session_id = entity_service.get_session_id()
            if not session_id:
                raise Exception('No session in progress')

            # Convert JSON to StrategyRequestEntity
            # strat_request = sync_to_async(json_to_strategy_request(strategy_data))
            # call with sync to async
            strat_request = await sync_to_async(self.json_to_strategy_request)(strategy_data)

            task = strategy_executor_service.execute_request(strat_request, wait=False)
            # Send confirmation of execution
            await self.send(json.dumps({
                'type': 'strategy_executed',
                'message': 'Strategy executed successfully'
            }))

        except Exception as e:
            print(f"Error executing strategy: {str(e)}")
            await self.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def entity_update(self, event):
        """Handle entity updates and send to clients"""
        try:
            await self.send(text_data=json.dumps({
                'type': 'entity_update',
                'entities': event['entities']
            }))
        except Exception as e:
            print(f"Error in entity_update: {str(e)}")

class EntityConsumer(AsyncWebsocketConsumer):
    """Handles individual entity updates"""

    async def connect(self):
        self.entity_id = self.scope['url_route']['kwargs']['entity_id']
        print(f"Connecting to Entity WebSocket for entity {self.entity_id}")

        await self.accept()
        
        await self.channel_layer.group_add(
            f"entity_{self.entity_id}",
            self.channel_name
        )

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            f"entity_{self.entity_id}",
            self.channel_name
        )

    async def receive(self, text_data):
        data = json.loads(text_data)
        command = data.get('command')

        if command == 'execute_strategy':
            await self.handle_execute_strategy(data.get('strategy_data', {}))
        else:
            await self.send(json.dumps({
                'type': 'error',
                'message': f'Unknown command: {command}'
            }))

    async def handle_execute_strategy(self, strategy_data):
        try:
            # Execute strategy and get updated entity data
            updated_entity = await sync_to_async(entity_service.execute_strategy)(
                self.entitythey_id,
                strategy_data
            )

            # Broadcast to entity-specific group
            await self.channel_layer.group_send(
                f"entity_{self.entity_id}",
                {
                    "type": "entity_update",
                    "entity": updated_entity.serialize()
                }
            )

            # Also broadcast to global group if there are related entity updates
            related_updates = updated_entity.get_related_updates()
            if related_updates:
                await self.channel_layer.group_send(
                    "global_entities",
                    {
                        "type": "entity_update",
                        "entities": related_updates
                    }
                )

        except Exception as e:
            await self.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def entity_update(self, event):
        """Handle entity updates and send to clients"""
        await self.send(text_data=json.dumps({
            'type': 'entity_update',
            'entity': event['entity']
        }))

