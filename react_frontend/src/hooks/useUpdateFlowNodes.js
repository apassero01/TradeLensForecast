import { useRecoilCallback } from 'recoil';
import { entityFamily } from '../state/entityFamily'; // Adjust import path if needed

/**
 * Hook that provides a function to update entity nodes in both Recoil state and React Flow
 */
const useUpdateFlowNodes = (reactFlowInstance) => {
  const updateEntity = useRecoilCallback(
    ({ set, snapshot }) => async (childId, updatedFields) => {
      console.log('updateEntity', childId, updatedFields);
      
      // First, make sure the node is visible (if we're trying to show it)
      if (updatedFields.hasOwnProperty('hidden') && updatedFields.hidden === false) {
        // First update React Flow nodes to make it visible
        reactFlowInstance.setNodes((nodes) => 
          nodes.map((node) => 
            node.id === childId 
              ? { ...node, hidden: false }
              : node
          )
        );
        
        // Give React Flow a moment to process this change
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      
      // Now update the Recoil state
      set(entityFamily(childId), (prev) => ({ ...prev, ...updatedFields }));
      
      // Wait for the atom update to propagate
      const updatedEntity = await snapshot.getPromise(entityFamily(childId));
      
      // Finally, ensure React Flow has the latest node data
      reactFlowInstance.setNodes((nodes) => 
        nodes.map((node) => 
          node.id === childId 
            ? {
                ...node,
                key: `${childId}-${Date.now()}`,
                hidden: updatedFields.hasOwnProperty('hidden') ? updatedFields.hidden : node.hidden,
                data: { 
                  ...node.data,
                  entityId: childId,
                  _updateTimestamp: Date.now() 
                }
              }
            : node
        )
      );
      
    },
    [reactFlowInstance]
  );

  return updateEntity;
};

export default useUpdateFlowNodes;