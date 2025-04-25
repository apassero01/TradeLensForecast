// HiddenSyncer.jsx
import { useEffect } from 'react';
import { useReactFlow } from '@xyflow/react';
import { useRecoilValue } from 'recoil';
import { allEntitiesSelector } from '../../state/entitiesSelectors';

export default function HiddenSyncer() {
  const flow = useReactFlow();            // ← always the real RF instance
  const allEntities = useRecoilValue(allEntitiesSelector);

  useEffect(() => {
    flow.setNodes((nds) =>
      nds.map((n) => {
        const ent = allEntities.find((e) => e.entity_id === n.id);
        if (ent) {
            if (ent.hidden !== n.hidden) {
                return { ...n, hidden: !!ent.hidden };
            }
        }
        return n;
      })
    );
  }, [flow, allEntities]);

  return null;
}