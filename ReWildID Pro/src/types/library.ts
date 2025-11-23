import { DBImage } from './electron';

export interface GroupData {
    id: number;
    name: string;
    created_at: number;
    images: DBImage[];
}

export interface DateSection {
    date: string; // YYYYMMDD
    groups: GroupData[];
}
