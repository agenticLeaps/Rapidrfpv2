import time
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
from ..config.settings import Config

@dataclass
class StepMetrics:
    """Metrics for a single processing step."""
    step_name: str
    start_time: float
    end_time: float
    duration: float
    status: str  # 'started', 'completed', 'failed'
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_steps: List['StepMetrics'] = field(default_factory=list)
    error_message: Optional[str] = None
    
    @property
    def duration_formatted(self) -> str:
        """Return duration in human-readable format."""
        if self.duration < 1:
            return f"{self.duration * 1000:.1f}ms"
        elif self.duration < 60:
            return f"{self.duration:.2f}s"
        else:
            minutes = int(self.duration // 60)
            seconds = self.duration % 60
            return f"{minutes}m {seconds:.1f}s"

@dataclass
class ProcessingSession:
    """Complete processing session with all steps."""
    session_id: str
    file_name: str
    file_size: int
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    status: str = 'started'  # 'started', 'completed', 'failed'
    steps: List[StepMetrics] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    @property
    def total_duration_formatted(self) -> str:
        """Return total duration in human-readable format."""
        if self.total_duration is None:
            return "In progress..."
        
        if self.total_duration < 1:
            return f"{self.total_duration * 1000:.1f}ms"
        elif self.total_duration < 60:
            return f"{self.total_duration:.2f}s"
        else:
            minutes = int(self.total_duration // 60)
            seconds = self.total_duration % 60
            return f"{minutes}m {seconds:.1f}s"

class PerformanceLogger:
    """Comprehensive performance logger for file processing pipeline."""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or os.path.join(Config.DATA_DIR, "performance_logs")
        self.current_session: Optional[ProcessingSession] = None
        self.current_step: Optional[StepMetrics] = None
        self.step_stack: List[StepMetrics] = []  # For nested steps
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(f'{__name__}.PerformanceLogger')
        
        # Set up performance log file
        self.performance_log_file = os.path.join(self.log_dir, "performance.log")
        self.json_log_file = os.path.join(self.log_dir, "performance_sessions.jsonl")
    
    def start_session(self, session_id: str, file_name: str, file_size: int) -> str:
        """Start a new processing session."""
        if self.current_session and self.current_session.status == 'started':
            self.logger.warning(f"Starting new session {session_id} while session {self.current_session.session_id} is still active")
        
        self.current_session = ProcessingSession(
            session_id=session_id,
            file_name=file_name,
            file_size=file_size,
            start_time=time.time()
        )
        
        self.logger.info(f"🚀 Started processing session: {session_id} | File: {file_name} ({file_size:,} bytes)")
        self._log_to_file(f"SESSION_START", {
            'session_id': session_id,
            'file_name': file_name,
            'file_size': file_size,
            'timestamp': datetime.now().isoformat()
        })
        
        return session_id
    
    @contextmanager
    def step(self, step_name: str, **metadata):
        """Context manager for tracking a processing step."""
        step_metrics = self._start_step(step_name, metadata)
        try:
            yield step_metrics
            self._end_step(step_metrics, 'completed')
        except Exception as e:
            self._end_step(step_metrics, 'failed', str(e))
            raise
    
    def _start_step(self, step_name: str, metadata: Dict[str, Any] = None) -> StepMetrics:
        """Start tracking a processing step."""
        start_time = time.time()
        
        step_metrics = StepMetrics(
            step_name=step_name,
            start_time=start_time,
            end_time=0,
            duration=0,
            status='started',
            metadata=metadata or {}
        )
        
        # Handle nested steps
        if self.current_step:
            self.step_stack.append(self.current_step)
        
        self.current_step = step_metrics
        
        # Add to current session
        if self.current_session:
            if self.step_stack:
                # This is a sub-step
                parent_step = self.step_stack[-1]
                parent_step.sub_steps.append(step_metrics)
            else:
                # This is a top-level step
                self.current_session.steps.append(step_metrics)
        
        self.logger.info(f"  📍 Started step: {step_name}")
        self._log_to_file(f"STEP_START", {
            'session_id': self.current_session.session_id if self.current_session else 'unknown',
            'step_name': step_name,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        })
        
        return step_metrics
    
    def _end_step(self, step_metrics: StepMetrics, status: str, error_message: str = None):
        """End tracking a processing step."""
        end_time = time.time()
        step_metrics.end_time = end_time
        step_metrics.duration = end_time - step_metrics.start_time
        step_metrics.status = status
        step_metrics.error_message = error_message
        
        status_icon = "✅" if status == 'completed' else "❌"
        duration_str = step_metrics.duration_formatted
        
        if status == 'completed':
            self.logger.info(f"  {status_icon} Completed step: {step_metrics.step_name} ({duration_str})")
        else:
            self.logger.error(f"  {status_icon} Failed step: {step_metrics.step_name} ({duration_str}) - {error_message}")
        
        self._log_to_file(f"STEP_END", {
            'session_id': self.current_session.session_id if self.current_session else 'unknown',
            'step_name': step_metrics.step_name,
            'duration': step_metrics.duration,
            'status': status,
            'error_message': error_message,
            'metadata': step_metrics.metadata,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update step tracking
        if self.step_stack:
            self.current_step = self.step_stack.pop()
        else:
            self.current_step = None
    
    def add_step_metadata(self, **metadata):
        """Add metadata to the current step."""
        if self.current_step and hasattr(self.current_step, 'metadata'):
            self.current_step.metadata.update(metadata)
    
    def end_session(self, status: str = 'completed', error_message: str = None) -> Optional[ProcessingSession]:
        """End the current processing session."""
        if not self.current_session:
            self.logger.warning("Attempting to end session but no session is active")
            return None
        
        end_time = time.time()
        self.current_session.end_time = end_time
        self.current_session.total_duration = end_time - self.current_session.start_time
        self.current_session.status = status
        self.current_session.error_message = error_message
        
        # Generate summary
        self._generate_session_summary()
        
        status_icon = "🎉" if status == 'completed' else "💥"
        duration_str = self.current_session.total_duration_formatted
        
        if status == 'completed':
            self.logger.info(f"{status_icon} Completed processing session: {self.current_session.session_id} ({duration_str})")
        else:
            self.logger.error(f"{status_icon} Failed processing session: {self.current_session.session_id} ({duration_str}) - {error_message}")
        
        # Log session summary
        self._log_session_summary()
        
        # Save to JSON file
        self._save_session_json()
        
        completed_session = self.current_session
        self.current_session = None
        self.current_step = None
        self.step_stack.clear()
        
        return completed_session
    
    def _generate_session_summary(self):
        """Generate summary statistics for the session."""
        if not self.current_session:
            return
        
        summary = {
            'total_steps': len(self.current_session.steps),
            'successful_steps': len([s for s in self.current_session.steps if s.status == 'completed']),
            'failed_steps': len([s for s in self.current_session.steps if s.status == 'failed']),
            'step_breakdown': {},
            'processing_rate': 0,  # bytes per second
        }
        
        # Calculate processing rate
        if self.current_session.total_duration and self.current_session.total_duration > 0:
            summary['processing_rate'] = self.current_session.file_size / self.current_session.total_duration
        
        # Step breakdown
        for step in self.current_session.steps:
            step_name = step.step_name
            if step_name not in summary['step_breakdown']:
                summary['step_breakdown'][step_name] = {
                    'duration': 0,
                    'percentage': 0,
                    'status': step.status,
                    'sub_steps': len(step.sub_steps)
                }
            
            summary['step_breakdown'][step_name]['duration'] = step.duration
            if self.current_session.total_duration:
                summary['step_breakdown'][step_name]['percentage'] = (step.duration / self.current_session.total_duration) * 100
        
        self.current_session.summary = summary
    
    def _log_session_summary(self):
        """Log detailed session summary."""
        if not self.current_session or not self.current_session.summary:
            return
        
        summary = self.current_session.summary
        self.logger.info(f"📊 Session Summary for {self.current_session.session_id}:")
        self.logger.info(f"   • Total Duration: {self.current_session.total_duration_formatted}")
        self.logger.info(f"   • File Size: {self.current_session.file_size:,} bytes")
        self.logger.info(f"   • Processing Rate: {summary['processing_rate']:.1f} bytes/sec")
        self.logger.info(f"   • Steps: {summary['successful_steps']}/{summary['total_steps']} successful")
        
        if summary['step_breakdown']:
            self.logger.info(f"   • Step Breakdown:")
            for step_name, breakdown in summary['step_breakdown'].items():
                status_icon = "✅" if breakdown['status'] == 'completed' else "❌"
                percentage = breakdown['percentage']
                duration = StepMetrics(step_name, 0, 0, breakdown['duration'], breakdown['status']).duration_formatted
                self.logger.info(f"     {status_icon} {step_name}: {duration} ({percentage:.1f}%)")
    
    def _log_to_file(self, event_type: str, data: Dict[str, Any]):
        """Log event to performance log file."""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                **data
            }
            
            with open(self.performance_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to write to performance log file: {e}")
    
    def _save_session_json(self):
        """Save complete session data to JSON Lines file."""
        if not self.current_session:
            return
        
        try:
            session_dict = asdict(self.current_session)
            
            with open(self.json_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(session_dict, default=str) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to save session JSON: {e}")
    
    def get_session_report(self, session_id: str = None) -> Dict[str, Any]:
        """Get formatted report for a session."""
        session = self.current_session if session_id is None else self._load_session(session_id)
        
        if not session:
            return {'error': 'Session not found'}
        
        return {
            'session_id': session.session_id,
            'file_info': {
                'name': session.file_name,
                'size_bytes': session.file_size,
                'size_formatted': self._format_file_size(session.file_size)
            },
            'timing': {
                'total_duration': session.total_duration,
                'total_duration_formatted': session.total_duration_formatted,
                'start_time': datetime.fromtimestamp(session.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(session.end_time).isoformat() if session.end_time else None
            },
            'status': session.status,
            'summary': session.summary,
            'steps': [
                {
                    'name': step.step_name,
                    'duration': step.duration,
                    'duration_formatted': step.duration_formatted,
                    'status': step.status,
                    'metadata': step.metadata,
                    'sub_steps_count': len(step.sub_steps),
                    'error_message': step.error_message
                }
                for step in session.steps
            ]
        }
    
    def _load_session(self, session_id: str) -> Optional[ProcessingSession]:
        """Load a session from the JSON log file."""
        try:
            with open(self.json_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    session_data = json.loads(line.strip())
                    if session_data.get('session_id') == session_id:
                        return ProcessingSession(**session_data)
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
        
        return None
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def export_session_report(self, session_id: str = None, output_path: str = None) -> str:
        """Export session report to a formatted text file."""
        report = self.get_session_report(session_id)
        
        if 'error' in report:
            return report['error']
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.log_dir, f"performance_report_{report['session_id']}_{timestamp}.txt")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"PERFORMANCE ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Session ID: {report['session_id']}\n")
                f.write(f"File: {report['file_info']['name']}\n")
                f.write(f"File Size: {report['file_info']['size_formatted']} ({report['file_info']['size_bytes']:,} bytes)\n")
                f.write(f"Status: {report['status']}\n")
                f.write(f"Start Time: {report['timing']['start_time']}\n")
                f.write(f"End Time: {report['timing']['end_time']}\n")
                f.write(f"Total Duration: {report['timing']['total_duration_formatted']}\n\n")
                
                if report['summary']:
                    f.write("SUMMARY\n")
                    f.write("-" * 40 + "\n")
                    summary = report['summary']
                    f.write(f"Processing Rate: {summary['processing_rate']:.1f} bytes/sec\n")
                    f.write(f"Total Steps: {summary['total_steps']}\n")
                    f.write(f"Successful Steps: {summary['successful_steps']}\n")
                    f.write(f"Failed Steps: {summary['failed_steps']}\n\n")
                
                f.write("STEP BREAKDOWN\n")
                f.write("-" * 40 + "\n")
                for step in report['steps']:
                    status_symbol = "✓" if step['status'] == 'completed' else "✗"
                    f.write(f"{status_symbol} {step['name']:<25} {step['duration_formatted']:>10}\n")
                    if step['sub_steps_count'] > 0:
                        f.write(f"  └─ Sub-steps: {step['sub_steps_count']}\n")
                    if step['error_message']:
                        f.write(f"  └─ Error: {step['error_message']}\n")
                    f.write("\n")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return f"Error: {e}"

# Global performance logger instance
performance_logger = PerformanceLogger()