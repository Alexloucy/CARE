import { useState } from 'react';
import { DateSection } from '../types/library';

export function useGroupActions(refreshLibrary: () => Promise<void> | void, dateSections: DateSection[]) {
    const [renameDialogOpen, setRenameDialogOpen] = useState(false);
    const [groupToRename, setGroupToRename] = useState<{ id: number, name: string } | null>(null);
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const [menuGroupId, setMenuGroupId] = useState<number | null>(null);

    const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, groupId: number) => {
        setAnchorEl(event.currentTarget);
        setMenuGroupId(groupId);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
        setMenuGroupId(null);
    };

    const handleDeleteGroup = async () => {
        if (menuGroupId === null) return;
        if (window.confirm('Are you sure you want to delete this group and all its images?')) {
            await window.api.deleteGroup(menuGroupId);
            await refreshLibrary();
        }
        handleMenuClose();
    };

    const handleRenameGroupClick = () => {
        if (menuGroupId === null) return;
        // Find group name
        let groupName = '';
        for (const section of dateSections) {
            const group = section.groups.find(g => g.id === menuGroupId);
            if (group) {
                groupName = group.name;
                break;
            }
        }
        setGroupToRename({ id: menuGroupId, name: groupName });
        setRenameDialogOpen(true);
        handleMenuClose();
    };

    const handleConfirmRename = async (newName: string) => {
        if (groupToRename) {
            await window.api.updateGroupName(groupToRename.id, newName);
            await refreshLibrary();
        }
        setRenameDialogOpen(false);
        setGroupToRename(null);
    };

    return {
        anchorEl,
        menuGroupId,
        renameDialogOpen,
        groupToRename,
        setRenameDialogOpen,
        setGroupToRename,
        handleMenuOpen,
        handleMenuClose,
        handleDeleteGroup,
        handleRenameGroupClick,
        handleConfirmRename
    };
}
