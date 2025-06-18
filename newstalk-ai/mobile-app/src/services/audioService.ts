import TrackPlayer, {
  AppKilledPlaybackBehavior,
  Capability,
  RepeatMode,
  Event,
} from 'react-native-track-player';

// TrackPlayer service setup
export const setupTrackPlayer = async () => {
  let isSetup = false;
  
  try {
    // Check if already setup
    const currentTrack = await TrackPlayer.getActiveTrack();
    isSetup = true;
  } catch {
    // Not setup yet
  }

  if (!isSetup) {
    await TrackPlayer.setupPlayer({
      maxCacheSize: 1024 * 10, // 10MB cache
    });

    await TrackPlayer.updateOptions({
      android: {
        appKilledPlaybackBehavior: AppKilledPlaybackBehavior.StopPlaybackAndRemoveNotification,
      },
      capabilities: [
        Capability.Play,
        Capability.Pause,
        Capability.SkipToNext,
        Capability.SkipToPrevious,
        Capability.SeekTo,
        Capability.Stop,
      ],
      compactCapabilities: [
        Capability.Play,
        Capability.Pause,
        Capability.SkipToNext,
        Capability.SkipToPrevious,
      ],
      progressUpdateEventInterval: 1,
    });

    await TrackPlayer.setRepeatMode(RepeatMode.Off);
  }
};

// Background playback service
export const setupPlayerService = async () => {
  await setupTrackPlayer();
  
  // Register playback service
  TrackPlayer.registerPlaybackService(() => require('./playbackService'));
};

// Playback controls
export const playAudio = async (track: {
  id: string;
  url: string;
  title: string;
  artist?: string;
  duration?: number;
  artwork?: string;
}) => {
  try {
    await TrackPlayer.reset();
    await TrackPlayer.add(track);
    await TrackPlayer.play();
  } catch (error) {
    console.error('Play audio error:', error);
    throw error;
  }
};

export const pauseAudio = async () => {
  try {
    await TrackPlayer.pause();
  } catch (error) {
    console.error('Pause audio error:', error);
    throw error;
  }
};

export const stopAudio = async () => {
  try {
    await TrackPlayer.stop();
    await TrackPlayer.reset();
  } catch (error) {
    console.error('Stop audio error:', error);
    throw error;
  }
};

export const seekTo = async (position: number) => {
  try {
    await TrackPlayer.seekTo(position);
  } catch (error) {
    console.error('Seek error:', error);
    throw error;
  }
};

export const setPlaybackRate = async (rate: number) => {
  try {
    await TrackPlayer.setRate(rate);
  } catch (error) {
    console.error('Set rate error:', error);
    throw error;
  }
};

// Audio session management
export const setupAudioSession = async () => {
  try {
    // Configure audio session for optimal playback
    // This would be platform-specific implementation
    console.log('Audio session configured');
  } catch (error) {
    console.error('Audio session setup error:', error);
  }
}; 