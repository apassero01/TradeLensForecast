// CustomContextMenu.js
import React from 'react';
import {
  ContextMenu,
  ContextMenuTrigger,
  MenuItem
} from 'react-contextmenu';

// This is a wrapper to unify some Tailwind styling
// If your canvas is scaled, you might need additional logic to offset the event coords
export const CustomContextMenuTrigger = ({ id, children }) => {
  return (
    <ContextMenuTrigger
      id={id}
      holdToDisplay={-1} // Right-click only
      attributes={{
        onContextMenu: (e) => {
          e.preventDefault();
        },
      }}
    >
      {children}
    </ContextMenuTrigger>
  );
};

export const CustomMenu = ({ id, onCopyId, onNewStrategy }) => {
  return (
    <ContextMenu
      id={id}
      className="react-contextmenu z-50 bg-gray-900 text-gray-200 border
                 border-gray-700 rounded shadow-lg py-1"
    >
      <MenuItem
        className="px-4 py-2 hover:bg-gray-700"
        onClick={onCopyId}
      >
        Copy ID
      </MenuItem>
      <MenuItem
        className="px-4 py-2 hover:bg-gray-700"
        onClick={onNewStrategy}
      >
        New Strategy
      </MenuItem>
    </ContextMenu>
  );
};