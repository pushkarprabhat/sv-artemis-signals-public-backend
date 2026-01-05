# utils/email_service.py
# Professional email service for reports and alerts
# Automated reporting and notification system

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from utils.logger import logger
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# EMAIL CONFIGURATION - Centralized Here
# Centralized email configuration management
# ============================================================================

# SMTP Server Settings
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# Sender Credentials (App Password, not regular password!)
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "pushkarprabhat@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")  # Gmail app password (16 chars)

# Backup email settings if Gmail fails
BACKUP_SMTP_SERVER = os.getenv("BACKUP_SMTP_SERVER", "smtp.outlook.com")
BACKUP_SMTP_PORT = int(os.getenv("BACKUP_SMTP_PORT", "587"))

# Email scheduling (IST times)
OPENING_BELL_TIME = "08:00"  # 8:00 AM IST
CLOSING_BELL_TIME = "17:55"  # 5:55 PM IST (5 min after market close)
DAILY_PNL_TIME = "18:00"  # 6:00 PM IST

# Default subscribers (will be loaded from CSV, fallback if CSV missing)
DEFAULT_SUBSCRIBERS = [
    {"email": "pushkarprabhat@gmail.com", "name": "Pushkar Prabhat"}
]

class EmailService:
    """
    Professional email service for automated reports
    
    Automated reporting: Sends daily reports to track market moves
    and system performance, ensuring continuous monitoring.
    """
    
    def __init__(self, 
                 smtp_server: str = None,
                 smtp_port: int = None,
                 sender_email: str = None,
                 sender_password: str = None):
        """
        Initialize email service
        
        Args:
            smtp_server: SMTP server address (defaults to config)
            smtp_port: SMTP port (defaults to config)
            sender_email: Sender's email address (defaults to config)
            sender_password: Sender's app password (defaults to config)
        """
        # Use provided values or fall back to config
        self.smtp_server = smtp_server or SMTP_SERVER
        self.smtp_port = smtp_port or SMTP_PORT
        self.sender_email = sender_email or SENDER_EMAIL
        self.sender_password = sender_password or SENDER_PASSWORD
        
        # Load subscription list
        self.subscriptions_file = Path("config/email_subscriptions.csv")
        self.load_subscriptions()
    
    def load_subscriptions(self):
        """Load email subscriptions from file"""
        try:
            if self.subscriptions_file.exists():
                self.subscriptions = pd.read_csv(self.subscriptions_file)
            else:
                # Create default with user's email
                # Use default subscribers from config
                default_emails = [s['email'] for s in DEFAULT_SUBSCRIBERS]
                default_names = [s['name'] for s in DEFAULT_SUBSCRIBERS]
                
                self.subscriptions = pd.DataFrame({
                    'email': default_emails,
                    'name': default_names,
                    'closing_bell': [True],
                    'opening_bell': [True],
                    'daily_pnl': [True],
                    'trade_alerts': [True],
                    'subscribed_date': [datetime.now().strftime('%Y-%m-%d')]
                })
                self.subscriptions_file.parent.mkdir(parents=True, exist_ok=True)
                self.subscriptions.to_csv(self.subscriptions_file, index=False)
                logger.info(f"Created subscription list with default email")
        except Exception as e:
            logger.error(f"Error loading subscriptions: {e}")
            self.subscriptions = pd.DataFrame()
    
    def add_subscriber(self, email: str, name: str, 
                      closing_bell: bool = True,
                      opening_bell: bool = True,
                      daily_pnl: bool = True,
                      trade_alerts: bool = True):
        """Add new subscriber to list"""
        try:
            new_sub = pd.DataFrame({
                'email': [email],
                'name': [name],
                'closing_bell': [closing_bell],
                'opening_bell': [opening_bell],
                'daily_pnl': [daily_pnl],
                'trade_alerts': [trade_alerts],
                'subscribed_date': [datetime.now().strftime('%Y-%m-%d')]
            })
            
            self.subscriptions = pd.concat([self.subscriptions, new_sub], ignore_index=True)
            self.subscriptions.to_csv(self.subscriptions_file, index=False)
            logger.info(f"Added subscriber: {email}")
            return True
        except Exception as e:
            logger.error(f"Error adding subscriber: {e}")
            return False
    
    def get_subscribers(self, report_type: str) -> List[str]:
        """Get list of emails subscribed to specific report type"""
        try:
            if self.subscriptions.empty:
                return []
            
            filtered = self.subscriptions[self.subscriptions[report_type] == True]
            return filtered['email'].tolist()
        except Exception as e:
            logger.error(f"Error getting subscribers for {report_type}: {e}")
            return []
    
    def send_email(self, 
                   recipient_email: str = None,
                   subject: str = None,
                   body_html: str = None,
                   attachments: Optional[List[Path]] = None,
                   body: str = None,
                   recipients: List[str] = None) -> bool:
        """
        Send professional HTML email with optional attachments
        
        Args:
            recipient_email: Single recipient's email (legacy)
            subject: Email subject
            body_html: HTML body content
            attachments: List of file paths to attach
            body: Plain text body content (will be wrapped in HTML if body_html missing)
            recipients: List of recipients (replaces recipient_email if provided)
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Handle multiple recipients or single one
            target_recipients = recipients or ([recipient_email] if recipient_email else [])
            if not target_recipients:
                logger.warning("No recipients specified for email")
                return False
                
            # Handle body vs body_html
            final_html = body_html or (f"<html><body><pre>{body}</pre></body></html>" if body else "<html><body></body></html>")
            
            success = True
            for email in target_recipients:
                # Create message
                msg = MIMEMultipart('alternative')
                msg['From'] = self.sender_email
                msg['To'] = email
                msg['Subject'] = subject
                
                # Attach HTML body
                html_part = MIMEText(final_html, 'html')
                msg.attach(html_part)
                
                # Attach files if provided
                if attachments:
                    for file_path in attachments:
                        if file_path.exists():
                            with open(file_path, 'rb') as f:
                                part = MIMEBase('application', 'octet-stream')
                                part.set_payload(f.read())
                                encoders.encode_base64(part)
                                part.add_header(
                                    'Content-Disposition',
                                    f'attachment; filename= {file_path.name}'
                                )
                                msg.attach(part)
                
                # Send email
                try:
                    with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                        server.starttls()
                        server.login(self.sender_email, self.sender_password)
                        server.send_message(msg)
                    logger.info(f"Email sent successfully to {email}: {subject}")
                except Exception as e:
                    logger.error(f"Error sending email to {email}: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error in send_email: {e}")
            return False
    
    def send_closing_bell_report(self, report_date: str = None):
        """Send Closing Bell Report to all subscribers"""
        if not report_date:
            report_date = datetime.now().strftime("%Y-%m-%d")
        
        subscribers = self.get_subscribers('closing_bell')
        
        if not subscribers:
            logger.warning("No subscribers for Closing Bell Report")
            return
        
        # Load report data
        report_file = Path(f"data/closing_bell_reports/closing_bell_{report_date}.csv")
        
        if not report_file.exists():
            logger.warning(f"Closing Bell Report not found for {report_date}")
            return
        
        # Generate HTML email
        subject = f"📊 Closing Bell Report - {report_date}"
        body_html = self._generate_closing_bell_html(report_file)
        
        # Send to all subscribers
        for email in subscribers:
            self.send_email(email, subject, body_html, [report_file])
    
    def send_daily_pnl_report(self, pnl_data: dict):
        """Send daily P&L and portfolio summary"""
        subscribers = self.get_subscribers('daily_pnl')
        
        if not subscribers:
            return
        
        subject = f"💰 Daily P&L Report - {datetime.now().strftime('%Y-%m-%d')}"
        body_html = self._generate_pnl_html(pnl_data)
        
        for email in subscribers:
            self.send_email(email, subject, body_html)
    
    def _generate_closing_bell_html(self, report_file: Path) -> str:
        """Generate professional HTML for Closing Bell Report"""
        try:
            df = pd.read_csv(report_file)
            
            # Calculate summary
            total_stocks = len(df)
            gainers = len(df[df['change_pct'] > 0]) if 'change_pct' in df.columns else 0
            losers = len(df[df['change_pct'] < 0]) if 'change_pct' in df.columns else 0
            
            # Top gainers/losers
            top_gainers = df.nlargest(5, 'change_pct')[['symbol', 'ltp', 'change_pct']].to_html(index=False)
            top_losers = df.nsmallest(5, 'change_pct')[['symbol', 'ltp', 'change_pct']].to_html(index=False)
            
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }}
                    .container {{ background-color: white; padding: 30px; border-radius: 10px; max-width: 800px; margin: 0 auto; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                    .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                    .metric {{ text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 8px; }}
                    .metric-value {{ font-size: 28px; font-weight: bold; color: #667eea; }}
                    .metric-label {{ color: #666; font-size: 14px; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #667eea; color: white; }}
                    .positive {{ color: #28a745; font-weight: bold; }}
                    .negative {{ color: #dc3545; font-weight: bold; }}
                    .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🔔 Closing Bell Report</h1>
                        <p>{datetime.now().strftime('%A, %B %d, %Y')}</p>
                    </div>
                    
                    <div class="summary">
                        <div class="metric">
                            <div class="metric-value">{total_stocks}</div>
                            <div class="metric-label">Total Stocks</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value positive">{gainers}</div>
                            <div class="metric-label">Gainers</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value negative">{losers}</div>
                            <div class="metric-label">Losers</div>
                        </div>
                    </div>
                    
                    <h2>📈 Top 5 Gainers</h2>
                    {top_gainers}
                    
                    <h2>📉 Top 5 Losers</h2>
                    {top_losers}
                    
                    <div class="footer">
                        {f'<p><em>Systematic execution brings us closer to our goals</em> 📊</p>' if not COMMERCIAL_MODE else '<p><em>Powered by quantitative excellence</em></p>'}
                        <p>Artemis Signals by ManekBaba | Powered by Discipline & Excellence</p>
                    </div>
                </div>
            </body>
            </html>
            """
            return html
        except Exception as e:
            logger.error(f"Error generating Closing Bell HTML: {e}")
            return "<html><body><h1>Error generating report</h1></body></html>"
    
    def _generate_pnl_html(self, pnl_data: dict) -> str:
        """Generate professional HTML for daily P&L report"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; max-width: 600px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .metric-large {{ font-size: 36px; font-weight: bold; margin: 20px 0; text-align: center; }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .neutral {{ color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>💰 Daily P&L Report</h1>
                    <p>{datetime.now().strftime('%A, %B %d, %Y')}</p>
                </div>
                
                <div class="metric-large {'positive' if pnl_data.get('total_pnl', 0) >= 0 else 'negative'}">
                    ₹{pnl_data.get('total_pnl', 0):,.2f}
                </div>
                
                <p><strong>Capital:</strong> ₹{pnl_data.get('capital', 0):,.2f}</p>
                <p><strong>Open Positions:</strong> {pnl_data.get('open_positions', 0)}</p>
                <p><strong>Closed Trades:</strong> {pnl_data.get('closed_trades', 0)}</p>
                
                <div style="text-align: center; margin-top: 30px; color: #666; font-size: 12px;">
                    {f'<p><em>Building wealth through systematic execution, one trade at a time</em> 📊</p>' if not COMMERCIAL_MODE else '<p><em>Systematic trading excellence</em></p>'}
                </div>
            </div>
        </body>
        </html>
        """
        return html


# Global instance
_email_service = None

def get_email_service() -> EmailService:
    """Get global email service instance"""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
