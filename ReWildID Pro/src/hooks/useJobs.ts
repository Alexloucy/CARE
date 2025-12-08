import { useEffect, useState } from 'react';

export interface Job {
    id: string;
    type: 'import' | 'thumbnail' | 'detect' | 'reid';
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    message: string;
    createdAt: number;
    error?: string;
}

export function useJobs() {
    const [jobs, setJobs] = useState<Job[]>([]);

    useEffect(() => {
        // Initial fetch
        window.api.getJobs().then(setJobs);

        // Listen for updates
        const removeListener = window.api.onJobUpdate((updatedJobs) => {
            setJobs(updatedJobs);
        });

        return () => {
            removeListener();
        };
    }, []);

    const cancelJob = (id: string) => window.api.cancelJob(id);
    const retryJob = (id: string) => window.api.retryJob(id);

    return { jobs, cancelJob, retryJob };
}
