import React, { useRef } from 'react';
import { Sparkle } from '@phosphor-icons/react';
import { useTheme } from '@mui/material';
import { DISPLACEMENT_MAP } from './liquidGlassMap';

interface Props {
    size?: number;
    icon?: React.ReactNode;
    onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
}

export const LiquidGlassButton: React.FC<Props> = ({
    size = 320,
    icon = <Sparkle size={48} weight="fill" />,
    onClick
}) => {
    const filterId = useRef(`frosted-${Math.random().toString(36).slice(2)}`);
    const theme = useTheme();
    const isLight = theme.palette.mode === 'light';

    return (
        <>
            <svg style={{ position: 'absolute', width: 0, height: 0 }}>
                <filter id={filterId.current} primitiveUnits="objectBoundingBox">
                    <feImage
                        href={DISPLACEMENT_MAP}
                        x="0" y="0" width="1" height="1"
                        result="map"
                    />
                    <feGaussianBlur in="SourceGraphic" stdDeviation="0.02" result="blur" />
                    <feDisplacementMap
                        in="blur"
                        in2="map"
                        scale="1"
                        xChannelSelector="R"
                        yChannelSelector="G"
                    />
                </filter>
            </svg>

            <button
                onClick={onClick}
                style={{
                    position: 'relative',
                    width: `${size}px`,
                    height: `${size}px`,
                    borderRadius: '50%',
                    // Keep a translucent "frosted" fill so background content shows through
                    background: isLight ? 'rgba(255,255,255,0.75)' : 'rgba(255,255,255,0.08)',
                    // Dark outline in light mode, light outline in dark mode
                    border: isLight ? '1.5px solid rgba(15,23,42,0.3)' : '1.5px solid rgba(255,255,255,0.6)',
                    boxShadow: isLight
                        ? '0 10px 22px rgba(15,23,42,0.22)'
                        : '0 16px 32px rgba(0,0,0,0.24)',
                    backdropFilter: `url(#${filterId.current})`,
                    WebkitBackdropFilter: `url(#${filterId.current})`,
                    display: 'grid',
                    placeItems: 'center',
                    cursor: 'pointer',
                    outline: 'none',
                    color: isLight ? 'rgba(15,23,42,0.9)' : 'white',
                    transition: 'transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease, border-color 0.15s ease'
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'scale(1.05)';
                    e.currentTarget.style.boxShadow = isLight
                        ? '0 14px 30px rgba(15,23,42,0.30)'
                        : '0 20px 40px rgba(0,0,0,0.3)';
                    e.currentTarget.style.background = isLight
                        ? 'rgba(255,255,255,0.65)'
                        : 'rgba(255,255,255,0.14)';
                    e.currentTarget.style.borderColor = isLight
                        ? 'rgba(15,23,42,0.9)'
                        : 'rgba(255,255,255,0.85)';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'scale(1)';
                    e.currentTarget.style.boxShadow = isLight
                        ? '0 10px 22px rgba(15,23,42,0.22)'
                        : '0 16px 32px rgba(0,0,0,0.24)';
                    e.currentTarget.style.background = isLight
                        ? 'rgba(255,255,255,0.45)'
                        : 'rgba(255,255,255,0.08)';
                    e.currentTarget.style.borderColor = isLight
                        ? 'rgba(15,23,42,0.7)'
                        : 'rgba(255,255,255,0.6)';
                }}
            >
                {icon}
            </button>
        </>
    );
};

export default LiquidGlassButton;
