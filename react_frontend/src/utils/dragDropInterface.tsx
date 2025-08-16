/**
 * Standardized Drag and Drop Interface for Entity Management
 * 
 * This interface provides a consistent way to handle drag and drop operations
 * across all components in the application. Every component that supports
 * dragging or dropping entities should use these utilities.
 */

import React from 'react';

// Standard data transfer types
export const DRAG_TYPES = {
  ENTITY_ID: 'application/x-entity-id',
  ENTITY_DATA: 'application/x-entity-data',
  SOURCE_CONTEXT: 'application/x-source-context',
} as const;

// Standard drag effects
export const DRAG_EFFECTS = {
  MOVE: 'move',
  COPY: 'copy',
  LINK: 'link',
} as const;

// Entity drag data interface
export interface EntityDragData {
  entityId: string;
  entityType?: string;
  sourceContext?: string;
  sourceParentId?: string;
  customData?: Record<string, any>;
}

// Drop zone configuration
export interface DropZoneConfig {
  acceptedTypes?: string[];
  dropEffect?: 'move' | 'copy' | 'link';
  onDrop: (data: EntityDragData) => void;
  onDragOver?: (event: DragEvent) => void;
  onDragEnter?: (event: DragEvent) => void;
  onDragLeave?: (event: DragEvent) => void;
}

/**
 * Utility class for standardized drag and drop operations
 */
export class EntityDragDropUtil {
  /**
   * Start dragging an entity with standardized data
   */
  static startDrag(
    event: React.DragEvent,
    dragData: EntityDragData,
    options: {
      dragEffect?: 'move' | 'copy' | 'link';
      dragImage?: HTMLElement;
      dragImageText?: string;
    } = {}
  ): void {
    const { dragEffect = 'move', dragImage, dragImageText } = options;

    // Set standard data
    event.dataTransfer.setData(DRAG_TYPES.ENTITY_ID, dragData.entityId);
    event.dataTransfer.setData(DRAG_TYPES.ENTITY_DATA, JSON.stringify(dragData));
    
    // Set source context if provided
    if (dragData.sourceContext) {
      event.dataTransfer.setData(DRAG_TYPES.SOURCE_CONTEXT, dragData.sourceContext);
    }

    // Set drag effect
    event.dataTransfer.effectAllowed = dragEffect;

    // Create custom drag image if text provided
    if (dragImageText && !dragImage) {
      const customDragImage = this.createDragImage(dragImageText);
      event.dataTransfer.setDragImage(customDragImage, 0, 0);
      // Clean up after drag starts
      setTimeout(() => document.body.removeChild(customDragImage), 0);
    } else if (dragImage) {
      event.dataTransfer.setDragImage(dragImage, 0, 0);
    }

    console.log('🎯 Entity drag started:', dragData);
  }

  /**
   * Handle drop with standardized data extraction
   */
  static handleDrop(
    event: React.DragEvent,
    onDrop: (data: EntityDragData) => void
  ): boolean {
    event.preventDefault();
    event.stopPropagation();

    const entityId = event.dataTransfer.getData(DRAG_TYPES.ENTITY_ID);
    
    if (!entityId) {
      console.warn('🚫 No entity ID found in drag data');
      return false;
    }

    // Try to get full entity data, fallback to just ID
    let dragData: EntityDragData = { entityId };
    
    try {
      const fullData = event.dataTransfer.getData(DRAG_TYPES.ENTITY_DATA);
      if (fullData) {
        dragData = JSON.parse(fullData);
      }
    } catch (error) {
      console.warn('⚠️ Could not parse full drag data, using entity ID only:', error);
    }

    console.log('🎯 Entity drop handled:', dragData);
    onDrop(dragData);
    return true;
  }

  /**
   * Setup standard drag over handler
   */
  static handleDragOver(
    event: React.DragEvent,
    options: {
      dropEffect?: 'move' | 'copy' | 'link';
      acceptedTypes?: string[];
    } = {}
  ): boolean {
    const { dropEffect = 'move', acceptedTypes } = options;

    // Check if we have valid entity data
    const entityId = event.dataTransfer.getData(DRAG_TYPES.ENTITY_ID);
    if (!entityId && acceptedTypes) {
      return false;
    }

    event.preventDefault();
    event.dataTransfer.dropEffect = dropEffect;
    return true;
  }

  /**
   * Create a custom drag image with text
   */
  private static createDragImage(text: string): HTMLElement {
    const dragImage = document.createElement('div');
    dragImage.style.position = 'absolute';
    dragImage.style.top = '-1000px';
    dragImage.style.backgroundColor = 'rgba(55, 65, 81, 0.9)';
    dragImage.style.color = 'white';
    dragImage.style.padding = '8px 12px';
    dragImage.style.borderRadius = '6px';
    dragImage.style.fontSize = '0.875rem';
    dragImage.style.fontWeight = '500';
    dragImage.style.border = '1px solid rgba(75, 85, 99, 0.5)';
    dragImage.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
    dragImage.textContent = text;
    document.body.appendChild(dragImage);
    return dragImage;
  }

  /**
   * Check if drag data contains a valid entity ID
   */
  static hasValidEntityData(event: React.DragEvent): boolean {
    return !!event.dataTransfer.getData(DRAG_TYPES.ENTITY_ID);
  }

  /**
   * Get entity ID from drag data without full parsing
   */
  static getEntityId(event: React.DragEvent): string | null {
    return event.dataTransfer.getData(DRAG_TYPES.ENTITY_ID) || null;
  }
}

/**
 * React hook for easy drag and drop setup
 */
export function useDragDrop(config: DropZoneConfig) {
  const handleDragOver = (event: React.DragEvent) => {
    const accepted = EntityDragDropUtil.handleDragOver(event, {
      dropEffect: config.dropEffect,
      acceptedTypes: config.acceptedTypes,
    });
    
    if (accepted && config.onDragOver) {
      config.onDragOver(event.nativeEvent);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    EntityDragDropUtil.handleDrop(event, config.onDrop);
  };

  const handleDragEnter = (event: React.DragEvent) => {
    if (EntityDragDropUtil.hasValidEntityData(event) && config.onDragEnter) {
      config.onDragEnter(event.nativeEvent);
    }
  };

  const handleDragLeave = (event: React.DragEvent) => {
    if (config.onDragLeave) {
      config.onDragLeave(event.nativeEvent);
    }
  };

  return {
    onDragOver: handleDragOver,
    onDrop: handleDrop,
    onDragEnter: handleDragEnter,
    onDragLeave: handleDragLeave,
  };
}

/**
 * Higher-order component to make any entity draggable
 */
export function makeDraggable<T extends { entityId: string }>(
  Component: React.ComponentType<T>,
  options: {
    getDragData?: (props: T) => EntityDragData;
    dragEffect?: 'move' | 'copy' | 'link';
    dragImageText?: (props: T) => string;
  } = {}
) {
  return function DraggableEntity(props: T) {
    const handleDragStart = (event: React.DragEvent) => {
      const dragData = options.getDragData 
        ? options.getDragData(props)
        : { entityId: props.entityId };

      EntityDragDropUtil.startDrag(event, dragData, {
        dragEffect: options.dragEffect,
        dragImageText: options.dragImageText?.(props),
      });
    };

    return (
      <div
        draggable
        onDragStart={handleDragStart}
        className="cursor-grab active:cursor-grabbing"
      >
        <Component {...props} />
      </div>
    );
  };
}
