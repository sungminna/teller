import TrackPlayer, { Event } from 'react-native-track-player';

module.exports = async function() {
  TrackPlayer.addEventListener(Event.RemotePlay, () => TrackPlayer.play());
  
  TrackPlayer.addEventListener(Event.RemotePause, () => TrackPlayer.pause());
  
  TrackPlayer.addEventListener(Event.RemoteStop, () => TrackPlayer.stop());
  
  TrackPlayer.addEventListener(Event.RemoteSeek, (event) => {
    TrackPlayer.seekTo(event.position);
  });
  
  TrackPlayer.addEventListener(Event.RemoteNext, () => {
    // Handle next track if implementing playlist
    console.log('Next track requested');
  });
  
  TrackPlayer.addEventListener(Event.RemotePrevious, () => {
    // Handle previous track if implementing playlist
    console.log('Previous track requested');
  });
  
  TrackPlayer.addEventListener(Event.PlaybackError, (event) => {
    console.error('Playback error:', event);
  });
  
  TrackPlayer.addEventListener(Event.PlaybackTrackChanged, (event) => {
    console.log('Track changed:', event);
  });
}; 