import { useCallback, useMemo } from 'react';

export interface ExtractedEntity {
    entity_id: string;
    entity_type: string;
    entity_name?: string;
    child_ids?: string[];
    parent_ids?: string[];
    [key: string]: any;
}

export interface EntityExtractionResult {
    entities: ExtractedEntity[];
    hasViews: boolean;
    totalEntities: number;
    entityTypes: string[];
}

/**
 * Custom hook for extracting and processing serialized entity data
 * from various sources like chat responses, JSON strings, etc.
 */
export const useEntityExtractor = () => {
    
    /**
     * Extract entity data from text content using multiple patterns
     */
    const extractEntityData = useCallback((content: string): any => {
        if (!content || typeof content !== 'string') {
            return null;
        }

        // Pattern 1: Look for Entity Graph section (from serialize_entities_and_strategies)
        const entityGraphMatch = content.match(/\s*={50}\r?\n\s*Entity Graph\r?\n\s*-{50}\r?\n(\s*\{[\s\S]*?\}\s*)\r?\n\s*={50}/);
        if (entityGraphMatch) {
            try {
                return JSON.parse(entityGraphMatch[1]);
            } catch (e) {
                console.warn('Failed to parse entity graph:', e, 'Captured content:', entityGraphMatch[1]);
            }
        }

        // Pattern 2: Look for serialized entity JSON blocks
        const entityJsonMatch = content.match(/```json\n([\s\S]*?)\n```/);
        if (entityJsonMatch) {
            try {
                const parsed = JSON.parse(entityJsonMatch[1]);
                if (isValidEntityData(parsed)) {
                    return parsed;
                }
            } catch (e) {
                console.warn('Failed to parse potential entity JSON:', e);
            }
        }

        // Pattern 3: Look for entities in StrategyRequest code blocks
        const strategyRequestMatch = content.match(/```StrategyRequest\n([\s\S]*?)\n```/);
        if (strategyRequestMatch) {
            try {
                const parsed = JSON.parse(strategyRequestMatch[1]);
                if (parsed.ret_val && parsed.ret_val.serialized_entities) {
                    return parsed.ret_val.serialized_entities;
                }
            } catch (e) {
                console.warn('Failed to parse strategy request:', e);
            }
        }

        // Pattern 4: Look for direct JSON objects that might be entities
        const jsonBlockMatches = content.matchAll(/\{[\s\S]*?\}/g);
        for (const match of jsonBlockMatches) {
            try {
                const parsed = JSON.parse(match[0]);
                if (isValidEntityData(parsed)) {
                    return parsed;
                }
            } catch (e) {
                // Continue to next match
            }
        }

        return null;
    }, []);

    /**
     * Check if data looks like valid entity data
     */
    const isValidEntityData = useCallback((data: any): boolean => {
        if (!data || typeof data !== 'object') {
            return false;
        }

        // Single entity
        if (data.entity_id && data.entity_type) {
            return true;
        }

        // Dictionary of entities
        if (typeof data === 'object' && !Array.isArray(data)) {
            const values = Object.values(data);
            if (values.length > 0) {
                const firstValue = values[0] as any;
                return firstValue && firstValue.entity_id && firstValue.entity_type;
            }
        }

        return false;
    }, []);

    /**
     * Process raw entity data into a standardized format
     */
    const processEntityData = useCallback((entityData: any): EntityExtractionResult => {
        if (!entityData) {
            return {
                entities: [],
                hasViews: false,
                totalEntities: 0,
                entityTypes: []
            };
        }

        // Convert to array format
        const entities: ExtractedEntity[] = entityData.entity_id 
            ? [entityData] 
            : Object.values(entityData);

        const validEntities = entities.filter(entity => 
            entity && entity.entity_id && entity.entity_type
        );

        const entityTypes = [...new Set(validEntities.map(e => e.entity_type))];
        const hasViews = validEntities.some(e => e.entity_type === 'view');

        return {
            entities: validEntities,
            hasViews,
            totalEntities: validEntities.length,
            entityTypes
        };
    }, []);

    /**
     * Find view children for a given entity within the extracted data
     */
    const findViewChildren = useCallback((entity: ExtractedEntity, allEntities: ExtractedEntity[]): ExtractedEntity[] => {
        if (!entity.child_ids || entity.child_ids.length === 0) {
            return [];
        }

        return allEntities.filter(e => 
            entity.child_ids!.includes(e.entity_id) && e.entity_type === 'view'
        );
    }, []);

    /**
     * Find the first view child for an entity
     */
    const findFirstViewChild = useCallback((entity: ExtractedEntity, allEntities: ExtractedEntity[]): ExtractedEntity | null => {
        const viewChildren = findViewChildren(entity, allEntities);
        return viewChildren.length > 0 ? viewChildren[0] : null;
    }, [findViewChildren]);

    /**
     * Get entities with their associated views
     */
    const getEntitiesWithViews = useCallback((entities: ExtractedEntity[]) => {
        return entities.map(entity => ({
            entity,
            viewChildren: findViewChildren(entity, entities),
            firstView: findFirstViewChild(entity, entities)
        }));
    }, [findViewChildren, findFirstViewChild]);

    /**
     * Extract and process entity data from content in one step
     */
    const extractAndProcess = useCallback((content: string): EntityExtractionResult => {
        const rawData = extractEntityData(content);
        return processEntityData(rawData);
    }, [extractEntityData, processEntityData]);

    return {
        extractEntityData,
        isValidEntityData,
        processEntityData,
        findViewChildren,
        findFirstViewChild,
        getEntitiesWithViews,
        extractAndProcess
    };
};

export default useEntityExtractor; 