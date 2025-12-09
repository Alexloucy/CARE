import { ReidRun, ReidIndividual } from './types';

// Simple in-memory cache for ReIDPage state
// This persists across navigation since it's a module-level singleton

interface ReidPageCache {
    runs: ReidRun[];
    individuals: Map<number, ReidIndividual[]>;
    pagination: Map<number, { page: number; hasMore: boolean }>;
    imageUrls: Map<string, string>;
    timestamp: number;
}

let cache: ReidPageCache | null = null;

// Cache expires after 5 minutes
const CACHE_TTL_MS = 5 * 60 * 1000;

// Track seen completed job IDs to detect new completions
const seenCompletedJobIds = new Set<string>();
let listenerInitialized = false;

// Initialize global listener for reid job completions
// This auto-invalidates the cache when new reid jobs complete
function initJobCompletionListener() {
    if (listenerInitialized || typeof window === 'undefined' || !window.api?.onJobUpdate) {
        return;
    }
    listenerInitialized = true;

    // First, mark all currently completed reid jobs as "seen"
    window.api.getJobs?.().then((jobs: { id: string; type: string; status: string }[]) => {
        jobs.forEach(job => {
            if (job.type === 'reid' && job.status === 'completed') {
                seenCompletedJobIds.add(job.id);
            }
        });
    }).catch(() => {
        // Ignore errors during initialization
    });

    // Listen for job updates
    window.api.onJobUpdate((jobs: { id: string; type: string; status: string }[]) => {
        let hasNewReidCompletion = false;

        jobs.forEach(job => {
            if (
                job.type === 'reid' &&
                job.status === 'completed' &&
                !seenCompletedJobIds.has(job.id)
            ) {
                seenCompletedJobIds.add(job.id);
                hasNewReidCompletion = true;
            }
        });

        // Auto-invalidate cache when a new reid job completes
        if (hasNewReidCompletion && cache) {
            cache = null;
        }
    });
}

// Initialize listener when module loads
initJobCompletionListener();

export function getReidPageCache(): ReidPageCache | null {
    if (!cache) return null;

    // Check if cache is still valid
    if (Date.now() - cache.timestamp > CACHE_TTL_MS) {
        cache = null;
        return null;
    }

    return cache;
}

export function setReidPageCache(data: Omit<ReidPageCache, 'timestamp'>): void {
    cache = {
        ...data,
        timestamp: Date.now()
    };
}

export function invalidateReidPageCache(): void {
    cache = null;
}

// Update specific parts of the cache without invalidating everything
export function updateReidPageCacheIndividuals(
    runId: number,
    individuals: ReidIndividual[],
    paginationData: { page: number; hasMore: boolean }
): void {
    if (!cache) return;

    const newIndividuals = new Map(cache.individuals);
    newIndividuals.set(runId, individuals);

    const newPagination = new Map(cache.pagination);
    newPagination.set(runId, paginationData);

    cache = {
        ...cache,
        individuals: newIndividuals,
        pagination: newPagination,
        timestamp: Date.now()
    };
}

export function updateReidPageCacheImageUrls(newUrls: Map<string, string>): void {
    if (!cache) return;

    const merged = new Map([...cache.imageUrls, ...newUrls]);
    cache = {
        ...cache,
        imageUrls: merged,
        timestamp: Date.now()
    };
}
