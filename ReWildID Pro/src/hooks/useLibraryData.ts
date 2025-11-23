import { useState, useEffect, useRef, useCallback } from 'react';
import { DateSection, GroupData } from '../types/library';

export interface LibraryFilterParams {
    date?: string | null;
    groupIds?: Set<number> | null;
    searchQuery?: string;
}

export function useLibraryData(filter?: LibraryFilterParams) {
    const [loading, setLoading] = useState(true);
    const [dateSections, setDateSections] = useState<DateSection[]>([]);
    const seenJobIds = useRef<Set<string>>(new Set());

    const fetchLibrary = useCallback(async () => {
        try {
            setLoading(true);
            
            const apiFilter = {
                date: filter?.date || undefined,
                groupIds: filter?.groupIds ? Array.from(filter.groupIds) : undefined,
                searchQuery: filter?.searchQuery || undefined
            };

            const response = await window.api.getImages(apiFilter);

            if (!response.ok || !response.images) {
                console.error('Failed to fetch library:', response.error);
                setLoading(false);
                return;
            }

            const images = response.images;
            const groupsMap: Record<number, GroupData> = {};

            // Group by Group ID first
            images.forEach(img => {
                if (!groupsMap[img.group_id]) {
                    groupsMap[img.group_id] = {
                        id: img.group_id,
                        name: img.group_name,
                        created_at: img.group_created_at,
                        images: []
                    };
                }
                groupsMap[img.group_id].images.push(img);
            });

            // Group by Date (using group_created_at)
            const dateMap: Record<string, GroupData[]> = {};

            Object.values(groupsMap).forEach(group => {
                // Use Local Time to match backend SQLite 'localtime' logic
                const d = new Date(group.created_at);
                const year = d.getFullYear();
                const month = String(d.getMonth() + 1).padStart(2, '0');
                const day = String(d.getDate()).padStart(2, '0');
                const date = `${year}${month}${day}`;

                if (!dateMap[date]) {
                    dateMap[date] = [];
                }
                dateMap[date].push(group);
            });

            // Sort Dates DESC, Groups DESC (by created_at)
            const sortedDates = Object.keys(dateMap).sort((a, b) => b.localeCompare(a));
            const newDateSections: DateSection[] = sortedDates.map(date => ({
                date,
                groups: dateMap[date].sort((a, b) => b.created_at - a.created_at)
            }));

            setDateSections(newDateSections);
        } catch (error) {
            console.error('Error loading library:', error);
        } finally {
            setLoading(false);
        }
    }, [filter?.date, filter?.groupIds, filter?.searchQuery]);

    useEffect(() => {
        fetchLibrary();
    }, [fetchLibrary]);

    // Listen for job completions to refresh library
    useEffect(() => {
        const removeListener = window.api.onJobUpdate((jobs) => {
            let shouldRefresh = false;
            jobs.forEach(job => {
                if (job.type === 'import' && job.status === 'completed') {
                    if (!seenJobIds.current.has(job.id)) {
                        seenJobIds.current.add(job.id);
                        shouldRefresh = true;
                    }
                }
            });
            
            if (shouldRefresh) {
                fetchLibrary();
            }
        });
        return removeListener;
    }, [fetchLibrary]);

    return { dateSections, loading, refreshLibrary: fetchLibrary };
}
