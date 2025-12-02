import React, { useRef } from 'react';
import { Sparkle } from '@phosphor-icons/react';
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
                    background: 'rgba(255,255,255,0.08)',
                    border: '2px solid transparent',
                    boxShadow: '0 0 0 2px rgba(255,255,255,0.6), 0 16px 32px rgba(0,0,0,0.12)',
                    backdropFilter: `url(#${filterId.current})`,
                    WebkitBackdropFilter: `url(#${filterId.current})`,
                    display: 'grid',
                    placeItems: 'center',
                    cursor: 'pointer',
                    outline: 'none',
                    color: 'white',
                    transition: 'transform 0.15s ease, box-shadow 0.15s ease'
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'scale(1.05)';
                    e.currentTarget.style.boxShadow = '0 0 0 2px rgba(255,255,255,0.8), 0 20px 40px rgba(0,0,0,0.2)';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'scale(1)';
                    e.currentTarget.style.boxShadow = '0 0 0 2px rgba(255,255,255,0.6), 0 16px 32px rgba(0,0,0,0.12)';
                }}
            >
                {icon}
            </button>
        </>
    );
};

export default LiquidGlassButton;
