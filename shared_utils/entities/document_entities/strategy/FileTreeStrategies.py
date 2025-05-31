from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.document_entities.DocumentEntity import DocumentEntity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
import os

class ScrapeFilePathStrategy(Strategy):
    """Strategy for scraping text content from a file path"""
    
    entity_type = EntityEnum.ENTITY  # Will be DOCUMENT once added
    strategy_description = 'Scrapes text content from a file or directory path'

    def verify_executable(self, entity: DocumentEntity, strategy_request: StrategyRequestEntity):
        """Verify the file path exists and is accessible"""
        config = strategy_request.param_config
        if 'file_path' not in config:
            raise ValueError("param_config must include 'file_path'")
            
        file_path = config['file_path']
        if not os.path.exists(file_path):
            raise ValueError(f"Path does not exist: {file_path}")
        
        return True

    def apply(self, entity: DocumentEntity) -> StrategyRequestEntity:
        """
        Scrape text from file path and set it on the document entity
        
        param_config requirements:
        - file_path: Path to file or directory to scrape
        """
        file_path = self.strategy_request.param_config['file_path']
        
        # Store the path
        entity.set_attribute('path', file_path)
        
        if os.path.isfile(file_path):
            # Read file content
            try:
                with open(file_path, 'r') as f:
                    text_content = f.read()
                entity.set_text(text_content)
                entity.set_document_type('file')
                entity.set_attribute('name', os.path.basename(file_path))
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                entity.set_document_type('file')
                entity.set_text(f"Error reading file: {e}")

        elif os.path.isdir(file_path):
            # Store directory info
            dir_name = os.path.basename(file_path)
            text_content = f"Directory: {dir_name}\nPath: {file_path}"
            entity.set_text(text_content)
            entity.set_document_type('directory')
            entity.set_attribute('name', dir_name)
            
        # Save changes
        self.strategy_request.ret_val['entity'] = entity
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'file_path': '',  # Path to file or directory to scrape
        } 
    

class GetFilePathWithDepth(Strategy):
    """Strategy for creating a document tree from a directory by discovering all paths first"""
    
    entity_type = EntityEnum.DOCUMENT
    strategy_description = 'Creates a document tree by discovering all paths first, then processing them'

    def should_process_path(self, path: str) -> bool:
        """Check if path should be processed (ignore hidden files/dirs and underscore files)"""
        basename = os.path.basename(path)
        return not (basename.startswith('.') or basename.startswith('_'))

    def discover_all_paths(self, root_path: str) -> list:
        """Discover all file and directory paths in the tree"""
        all_paths = []
        
        # Walk through the directory tree
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Filter out hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and not d.startswith('_')]
            
            # Add the current directory if it should be processed
            if self.should_process_path(dirpath):
                all_paths.append(dirpath)
            
            # Add all files in this directory
            for filename in filenames:
                if not filename.startswith('.') and not filename.startswith('_'):
                    file_path = os.path.join(dirpath, filename)
                    all_paths.append(file_path)
        
        return sorted(all_paths)  # Sort to ensure consistent ordering

    def get_parent_path(self, path: str, all_paths: list) -> str:
        """Find the immediate parent directory path from the list of all paths"""
        parent_dir = os.path.dirname(path)
        
        # Find the parent in our list of paths
        while parent_dir:
            if parent_dir in all_paths:
                return parent_dir
            parent_dir = os.path.dirname(parent_dir)
        
        return None

    def apply(self, entity: DocumentEntity) -> StrategyRequestEntity:
        """
        Create a document tree by first discovering all paths, then processing them
        
        param_config requirements:
        - root_path: Root directory path to create tree from
        """

        for child in entity.get_children():
            remove_child_request = StrategyRequestEntity()
            remove_child_request.strategy_name = 'RemoveEntityStrategy'
            remove_child_request.target_entity_id = child
            self.executor_service.execute_request(remove_child_request)

        entity = self.entity_service.get_entity(entity.entity_id)
        root_path = self.strategy_request.param_config['root_path']
        
        # First, discover all paths
        all_paths = self.discover_all_paths(root_path)
        
        # Create a mapping from path to entity
        path_to_entity = {}
        
        # Process the root entity first
        scrape_request = StrategyRequestEntity()
        scrape_request.strategy_name = 'ScrapeFilePathStrategy'
        scrape_request.param_config = {'file_path': root_path}
        scrape_request.target_entity_id = entity.entity_id
        scrape_response = self.executor_service.execute_request(scrape_request)
        entity = self.entity_service.get_entity(scrape_request.target_entity_id)
        path_to_entity[root_path] = entity
        
        # Process all other paths
        for path in all_paths:
            if path == root_path:
                continue  # Already processed
            
            # Find the parent entity
            parent_path = self.get_parent_path(path, all_paths)
            parent_entity = path_to_entity.get(parent_path, entity)  # Default to root if no parent found
            
            # Create new document entity as child
            create_request = StrategyRequestEntity()
            create_request.strategy_name = 'CreateEntityStrategy'
            create_request.param_config = {
                'entity_class': 'shared_utils.entities.document_entities.DocumentEntity.DocumentEntity',
                'initial_attributes': {
                    'hidden': True,
                }
            }
            create_request.target_entity_id = parent_entity.entity_id
            
            response = self.executor_service.execute_request(create_request)
            child_entity_id = response.ret_val['child_entity'].entity_id
            child_entity = self.entity_service.get_entity(child_entity_id)
            self.strategy_request.add_nested_request(create_request)
            parent_entity = self.entity_service.get_entity(parent_entity.entity_id)
            path_to_entity[parent_path] = parent_entity
            
            # Scrape the file/directory content
            scrape_request = StrategyRequestEntity()
            scrape_request.strategy_name = 'ScrapeFilePathStrategy'
            scrape_request.param_config = {'file_path': path}
            scrape_request.target_entity_id = child_entity.entity_id
            scrape_response = self.executor_service.execute_request(scrape_request)
            child_entity = self.entity_service.get_entity(child_entity.entity_id)
            self.strategy_request.add_nested_request(scrape_response)
            
            # Store in our mapping
            path_to_entity[path] = child_entity
            
            # Save the child entity
            self.entity_service.save_entity(child_entity)

        # Return the strategy request with all nested operations
        self.strategy_request.ret_val['total_paths_processed'] = len(all_paths)
        self.entity_service.save_entity(path_to_entity[root_path])  # Save the root entity after processing
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'root_path': '',  # Root directory path to create tree from
        }