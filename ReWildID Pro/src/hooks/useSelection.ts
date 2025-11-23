import { useState, useCallback } from 'react';

export function useSelection<T>(initialMode = false) {
    const [isSelectionMode, setIsSelectionMode] = useState(initialMode);
    const [selectedIds, setSelectedIds] = useState<Set<T>>(new Set());

    const toggleSelectionMode = useCallback(() => {
        setIsSelectionMode(prev => {
            if (prev) setSelectedIds(new Set()); // Clear on exit
            return !prev;
        });
    }, []);

    const toggleItem = useCallback((id: T) => {
        setSelectedIds(prev => {
            const newSet = new Set(prev);
            if (newSet.has(id)) {
                newSet.delete(id);
            } else {
                newSet.add(id);
            }
            return newSet;
        });
    }, []);

    const clearSelection = useCallback(() => {
        setSelectedIds(new Set());
        setIsSelectionMode(false);
    }, []);

    const setSelection = useCallback((ids: Set<T>) => {
        setSelectedIds(ids);
    }, []);

    return {
        isSelectionMode,
        selectedIds,
        toggleSelectionMode,
        toggleItem,
        clearSelection,
        setSelection,
        setIsSelectionMode
    };
}
