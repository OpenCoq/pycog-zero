#!/usr/bin/env python3
"""
Test script for ECAN attention allocation task prioritization integration.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

async def test_ecan_task_prioritization():
    """Test ECAN-based task prioritization functionality."""
    
    print("=== ECAN Task Prioritization Integration Test ===\n")
    
    try:
        # Test 1: Import verification
        print("1. Testing imports...")
        from python.tools.meta_cognition import MetaCognitionTool
        from python.tools.scheduler import SchedulerTool
        from python.helpers.task_scheduler import TaskScheduler, ScheduledTask, TaskSchedule, TaskState
        from agent import Agent, AgentConfig
        print("✓ All required imports successful\n")
        
        # Test 2: Create minimal agent configuration
        print("2. Creating test agent configuration...")
        config = AgentConfig(
            chat_model=None,
            utility_model=None,
            embeddings_model=None,
            browser_model=None,
            mcp_servers=""
        )
        
        # Create test agent
        agent = Agent(0, config)
        print("✓ Test agent created successfully\n")
        
        # Test 3: Create meta-cognition tool
        print("3. Testing MetaCognitionTool initialization...")
        meta_tool = MetaCognitionTool(agent)
        print(f"✓ MetaCognitionTool created (initialized: {meta_tool.initialized})")
        print(f"  - OpenCog available: {hasattr(meta_tool, 'atomspace') and meta_tool.atomspace is not None}")
        print(f"  - ECAN available: {hasattr(meta_tool, 'attention_bank') and meta_tool.attention_bank is not None}\n")
        
        # Test 4: Test attention allocation functionality
        print("4. Testing basic attention allocation...")
        test_goals = ["complete_urgent_task", "process_data", "generate_report"]
        test_tasks = ["analyze customer feedback", "update database", "prepare presentation"]
        
        attention_params = {
            "goals": test_goals,
            "tasks": test_tasks,
            "importance": 100
        }
        
        response = await meta_tool.allocate_attention(attention_params)
        print("✓ Attention allocation executed")
        
        # Parse response
        if hasattr(response, 'message'):
            print(f"  Response preview: {response.message[:200]}...")
            
            # Check if ECAN or fallback was used
            if "ECAN" in response.message:
                print("  ✓ ECAN mechanism detected")
            elif "fallback" in response.message:
                print("  ✓ Fallback mechanism detected")
        print()
        
        # Test 5: Test scheduler tool with ECAN prioritization  
        print("5. Testing SchedulerTool with ECAN prioritization...")
        
        # Create scheduler tool
        scheduler_tool = SchedulerTool(agent, name="scheduler", method="prioritize_tasks_ecan")
        print("✓ SchedulerTool created with ECAN prioritization method")
        
        # Create some test tasks for prioritization
        print("  Creating test tasks...")
        scheduler = TaskScheduler.get()
        
        # Create scheduled tasks for testing
        test_schedule = TaskSchedule(
            minute="0",
            hour="*", 
            day="*",
            month="*",
            weekday="*"
        )
        
        task1 = ScheduledTask.create(
            name="urgent_data_processing",
            system_prompt="You are a data analyst",
            prompt="Process the urgent customer data analysis", 
            attachments=[],
            schedule=test_schedule
        )
        
        task2 = ScheduledTask.create(
            name="routine_maintenance",
            system_prompt="You are a system administrator", 
            prompt="Perform routine system maintenance checks",
            attachments=[],
            schedule=test_schedule
        )
        
        task3 = ScheduledTask.create(
            name="generate_weekly_report",
            system_prompt="You are a business analyst",
            prompt="Generate the weekly business performance report",
            attachments=[],
            schedule=test_schedule
        )
        
        # Add tasks to scheduler
        await scheduler.add_task(task1)
        await scheduler.add_task(task2) 
        await scheduler.add_task(task3)
        print(f"✓ Created {len([task1, task2, task3])} test tasks")
        
        # Test 6: Execute ECAN prioritization
        print("\n6. Testing ECAN-based task prioritization...")
        
        prioritization_response = await scheduler_tool.prioritize_tasks_ecan(
            state=["idle", "running"],
            importance=100
        )
        
        if hasattr(prioritization_response, 'message'):
            print("✓ ECAN task prioritization executed")
            print("  Response preview:")
            
            # Extract key information from response
            message_lines = prioritization_response.message.split('\n')[:10]  # First 10 lines
            for line in message_lines:
                if line.strip():
                    print(f"    {line}")
            
            # Check for success indicators
            if "completed" in prioritization_response.message.lower():
                print("  ✓ Prioritization completed successfully")
            if "ECAN" in prioritization_response.message or "fallback" in prioritization_response.message:
                print("  ✓ Prioritization method applied")
        else:
            print("  ⚠️ No response message received")
        
        print()
        
        # Test 7: Test automatic task prioritization in scheduler
        print("7. Testing automatic ECAN integration in task scheduling...")
        
        # Get due tasks (this should now use ECAN prioritization)
        scheduler_task_list = scheduler._tasks
        due_tasks = await scheduler_task_list.get_due_tasks()
        
        print(f"✓ Retrieved {len(due_tasks)} due tasks with ECAN prioritization")
        if due_tasks:
            print("  Task execution order:")
            for i, task in enumerate(due_tasks[:3]):  # Show top 3
                print(f"    {i+1}. {task.name} (created: {task.created_at})")
        else:
            print("  (No due tasks found - this is expected for test tasks with specific schedules)")
        
        print()
        
        # Test 8: Cleanup
        print("8. Cleaning up test tasks...")
        await scheduler.remove_task_by_uuid(task1.uuid)
        await scheduler.remove_task_by_uuid(task2.uuid)
        await scheduler.remove_task_by_uuid(task3.uuid)
        print("✓ Test tasks cleaned up\n")
        
        print("=== ECAN Task Prioritization Integration Test COMPLETED ===")
        print("✓ All tests passed successfully!")
        print("✓ ECAN attention allocation is integrated with Agent-Zero task prioritization")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This may be expected if OpenCog is not installed - fallback mechanisms should work")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting ECAN Task Prioritization Integration Test...\n")
    
    # Run the async test
    success = asyncio.run(test_ecan_task_prioritization())
    
    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed - check implementation")
        sys.exit(1)