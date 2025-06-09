import React, { useEffect } from 'react';
import EntityNodeBase from './EntityNodeBase';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../state/entitiesSelectors';
import { EntityTypes } from './EntityEnum';
import useRenderStoredView from '../../../hooks/useRenderStoredView';

function DocumentEntity({ data, updateEntity, sendStrategyRequest }) {

  // console.log('DocumentEntity', EntityTypes.VIEW);
  const viewEntities = useRecoilValue(childrenByTypeSelector({ parentId: data.entityId, type: EntityTypes.VIEW }));

  const editorViewEntity = viewEntities.find(view =>
    view.data?.view_component_type === "advanced_document_editor"
  );

  const editorViewEntityIdToUse = editorViewEntity?.entity_id;

  const editorView = useRenderStoredView(editorViewEntityIdToUse, sendStrategyRequest, updateEntity);

  const renderContent = () => (
    <div className="flex flex-col h-full w-full">
      <h4 className="text-xs text-gray-500 mt-2 mb-1 px-2 flex-shrink-0">Editor View:</h4>
      <div className="nodrag border border-gray-300 rounded p-1 m-1 bg-gray-50 flex-grow min-h-0 overflow-auto">
        {editorView ? editorView : <p className="text-xs text-gray-400 italic">No editor view available.</p>}
      </div>
    </div>
  );

  return (
    <EntityNodeBase
      data={data}
      updateEntity={updateEntity}
    >
      {renderContent}
    </EntityNodeBase>
  );
}

export default React.memo(DocumentEntity);
